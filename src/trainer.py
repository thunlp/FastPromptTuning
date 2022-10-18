# This file defines the trainer of FPT methods. It defines the training and inference process
# Author: Yufei Huang
# Date: 2022-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import math
import copy
import torch
import logging
import os
import random
import math
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)
from transformers import (
    AdamW,
    Adafactor,
    get_scheduler,
    is_torch_available,
)
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import itertools
from torch.utils.tensorboard import SummaryWriter
from prompt_init import uniform_init, embedding_init
from utiles import SELECT_METHOD, get_attributes, NEURON_METHOD, load_tensor_from_csv
from t5.modeling_t5 import T5DenseGatedGeluDense
from tqdm import tqdm

logger = logging.getLogger('trainer')

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

class Trainer:
    '''
        Defines the training process and different progressive methods.
    '''
    DEVICE_MAPS = {
        2: {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        },
        4: {
            0: [0, 1, 2],
            1: [3, 4, 5, 6, 7, 8, 9],
            2: [10, 11, 12, 13, 14, 15, 16],
            3: [17, 18, 19, 20, 21, 22, 23]
        }
    }
    def __init__(self, args, model_provider, dataset_provider):
        self.args = args
        if args.seed is not None:
            set_seed(args.seed)
        logger.info("Loading model ...")
        self._model, self.config, self.tokenizer = model_provider(args)
        # for progressive learning, memory for some necessary information
        self.current_config = copy.deepcopy(self.config)   # current config for pruned model
        self.layers_index = {'encoder': list(range(self.current_config.num_layers)),
                             'decoder': list(range(self.current_config.num_decoder_layers))}  # curent layers be used
        self.device = torch.device('cpu')
        self.buffer = None
        if args.model_parallel_size > 1:
            self.device_maps = self.DEVICE_MAPS[args.model_parallel_size]
        self.device = self.init_device(args)
        if args.model_parallel_size > 1:
            self.current_device_maps = {
                'encoder': copy.deepcopy(self.device_maps),
                'decoder': copy.deepcopy(self.device_maps)
            }
        if args.early_exit:
            if args.decoder_keep_layers == 'all':
                logger.info("All layers in decoder will be kept.")
            else:
                logger.info(f"Prune decoder layers, only keep {args.decoder_keep_layers} layer in decoder.")
                decoder_layers = list(map(int, args.decoder_keep_layers.split(',')))
                self.prune_decoder_layers(args, decoder_layers, self._model)
            if args.encoder_keep_layers == 'all':
                logger.info("All layers in encoder will be kept.")
            else:
                logger.info(f"Prune encoder layers, only keep {args.encoder_keep_layers} layer in encoder.")
                encoder_layers = list(map(int, args.encoder_keep_layers.split(',')))
                self.prune_encoder_layers(args, encoder_layers, self._model)
        self.layer_scores = load_tensor_from_csv(args.score_file_layer)
        if args.encoder_progressive_learning:
            num_layers = self.config.num_layers
            init_num_layers = int(args.progressive_layers.split(',')[0])
            encoder_layers = SELECT_METHOD[args.select_layer_method](init_num_layers, num_layers,
                                                                     scores=self.layer_scores)
            self.full_encoder = self.prune_encoder_layers(args, encoder_layers, self._model)
            logger.info(f"Encoder progressive learning, initial layer is {str(encoder_layers)} ...")
            self.current_config.num_layers = init_num_layers
            self.layers_index['encoder'] = encoder_layers
            if self.args.model_parallel_size > 1:
                self.current_device_maps['encoder'] = self.modify_device_maps(self.device_maps, self.layers_index['encoder'])
        if args.decoder_progressive_learning:
            num_encoder_layers = self.config.num_layers
            num_layers = self.config.num_decoder_layers
            init_num_layers = int(args.progressive_layers.split(',')[0])
            decoder_layers = SELECT_METHOD[args.select_layer_method](init_num_layers, num_layers,
                                                                     scores=self.layer_scores,
                                                                     num_encoder_layers=num_encoder_layers)
            self.full_decoder = self.prune_decoder_layers(args, decoder_layers, self._model)
            logger.info(f"Decoder progressive learning, initial layer is {str(decoder_layers)} ...")
            self.current_config.num_decoder_layers = init_num_layers
            self.layers_index['decoder'] = decoder_layers
            if self.args.model_parallel_size > 1:
                self.current_device_maps['decoder'] = self.modify_device_maps(self.device_maps, self.layers_index['decoder'])
        if args.encoder_ffn_progressive or args.decoder_ffn_progressive:
            self.full_layers = {
                'encoder': [None] * self.config.num_layers,
                'decoder': [None] * self.config.num_decoder_layers,
            }
            self.scores = load_tensor_from_csv(args.score_file)
            ffn_progressive_dimentions = list(map(int, self.args.progressive_ffn_dimentions.split(',')))
            self.modify_ffn_layers(args, self._model, ffn_progressive_dimentions[0])
            logger.info(f"Modify ffn layers by {args.select_ffn_method}")
            if args.encoder_ffn_progressive:
                logger.info(f"Modify ffn layers in encoder, initial dimension is {ffn_progressive_dimentions[0]}")
            if args.decoder_ffn_progressive:
                logger.info(f"Modify ffn layers in decoder, initial dimension is {ffn_progressive_dimentions[0]}")
        logger.info("Loading Dataset ...")
        self.train_dataset, self.valid_dataset, self.test_dataset = dataset_provider(args, self.tokenizer)
        # self.valid_dataset = self.train_dataset  # debug
        max_target_length = 1
        # if self.args.max_target_length < 512:
        #     if self.train_dataset is not None:
        #         max_target_length = self.train_dataset.get_max_target_length()
        #     elif self.valid_dataset is not None:
        #         max_target_length = self.valid_dataset.get_max_target_length()
        if args.max_target_length < max_target_length:
            logger.info(f"Max Target Length less than data in dataset. "
                        f"Change from {args.max_target_length} to {min(max_target_length, 512)}")
            args.max_target_length = min(max_target_length, 512)
            self.set_dataset_max_target_length([self.train_dataset, self.valid_dataset, self.test_dataset], args.max_target_length)
        self.rank = args.rank
        self.world_size = args.world_size
        self.data_parallel_size = args.data_parallel_size
        self.move_to_device(self.device)
        self.wrape_model = None
        self.gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * args.data_parallel_size)
        self.init_tensorboard(args)
        if args.fp16:
            self.scaler = GradScaler()
        if args.tune_method == 'prompt':
            self.prompt = torch.rand((args.prompt_length, self.config.d_model), requires_grad=True, device=self.device)
            self.prepare_data = self.prepare_prompt_data
            if args.prompt_init == 'uniform':
                self.prompt = uniform_init(prompt=self.prompt, a=-math.sqrt(1 / self.config.d_model), b=math.sqrt(1 / self.config.d_model))
                # self.prompt = uniform_init(prompt=self.prompt, a=-0.5, b=0.5)
            elif args.prompt_init == 'embedding':
                self.prompt = embedding_init(self.prompt, self._model.get_input_embeddings().weight)
        else:
            self.prepare_data = self.prepare_model_data
    
    @property
    def model(self):
        if self.wrape_model is not None:
            return self.wrape_model
        else:
            # if self.args.model_parallel_size > 1:
            #     device_map = self.device_maps
            #     assert self.args.model_parallel_size == len(device_map)
            #     self._model.parallelize(device_map, )
            #     # logger.info("Model parallel {} part by {}".format(self.args.model_parallel_size, str(device_map)))
            if self.data_parallel_size > 1 and self.args.tune_method == 'model':
                device_ids = list(range(self.rank * self.args.model_parallel_size,
                                        (self.rank + 1) * self.args.model_parallel_size))
                self.wrape_model = DDP(self._model, device_ids=device_ids,
                                       output_device=self.rank * self.args.model_parallel_size)
            else:
                self.wrape_model = self._model
        return self.wrape_model

    def init_device(self, args):
        if args.model_parallel_size > 1:
            new_device_map = {}
            for key in self.device_maps:
                new_device_map[int(key) * args.data_parallel_size + args.rank] = self.device_maps[key]
            self.device_maps = new_device_map
        if args.no_cuda or (not torch.cuda.is_available()):
            return torch.device('cpu')
        else:
            return torch.device(f"cuda:{args.rank}")

    def move_to_device(self, device):
        if dist.is_initialized():
            torch.distributed.barrier()
        if self.args.model_parallel_size == 1:
            self._model = self._model.to(device)
        else:
            device_map = self.device_maps
            logger.warning("Rank {}: Device map: {}".format(self.args.rank, str(device_map)))
            assert self.args.model_parallel_size == len(device_map)
            self._model.parallelize(self.current_device_maps['encoder'], self.current_device_maps['decoder'])
            logger.warning("Rank {}: Model parallel {} part.".format(self.args.rank, self.args.model_parallel_size))
            if dist.is_initialized():
                torch.distributed.barrier()
    
    def init_tensorboard(self, args):
        self.tensorboard = None
        if args.tensorboard_dir is not None and self.rank == 0:
            self.tensorboard = SummaryWriter(log_dir=args.tensorboard_dir)

    def get_optimzied_group(self):
        size = 0
        if self.args.tune_method == 'model':
            no_decay = ["bias", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            to_update = self.model.parameters()
        else:
            for n, p in self.model.named_parameters():
                p.requires_grad = False
            optimizer_grouped_parameters = [
                {
                    "params": [self.prompt],
                    "weight_decay": self.args.weight_decay,
                }
            ]
            to_update = [self.prompt]
            if self.args.add_cross_ffn:
                current_optimizer_group = {
                    "params": [],
                    "weight_decay": 0.0,
                    "lr": self.args.cross_ffn_lr,
                }
                for n, p in self.model.named_parameters():
                    if 'cross_ffn' in n:
                        p.requires_grad = True
                        to_update.append(p)
                        current_optimizer_group['params'].append(p)
                optimizer_grouped_parameters.append(current_optimizer_group)
        for p in to_update:
            size += p.numel()
        return optimizer_grouped_parameters, to_update, size

    def train(self):
        # defines the train process
        if not os.path.exists(self.args.save) and self.rank == 0:
            os.mkdir(self.args.save)
        if not self.args.debug:
            self.model.train()
        else:
            self.model.eval()
        train_dataloader, train_sampler = self.build_dataloader(self.args, self.train_dataset,
                          self.args.micro_batch_size)
        if self.args.train_iters is None:
            self.args.train_iters = (
                                    len(train_dataloader) // self.gradient_accumulation_steps
                                    * float(self.args.train_epochs)
                                )
        if self.args.train_epochs is None:
            self.args.train_epochs = (self.args.train_iters * self.gradient_accumulation_steps) \
                                     // len(train_dataloader) + 1
        if self.args.lr_decay_iters is None:
            self.args.lr_decay_iters = self.args.train_iters
        # set progressive interval for progressive learning
        if self.args.decoder_progressive_learning or self.args.encoder_progressive_learning:
            progressive_layers = list(map(int, self.args.progressive_layers.split(',')))
            progressive_num = len(progressive_layers)
            progressive_intervals = list(map(int, self.args.progressive_train_iters.split(',')))
            cnt = 0
            progressive_updates = []
            for interval in progressive_intervals:
                cnt += interval
                progressive_updates.append(cnt)
            assert len(progressive_updates) == len(progressive_layers)
            progressive_layers.append(self.config.num_layers)
            progressive_updates.append(self.args.train_iters)
            max_progressive_updates = progressive_updates[-2]
            progressive_index = 0
            logger.info(f"Progressive learning on layers: increase model after {str(progressive_updates)} updates.")
            # set progressive interval for progressive learning
        if self.args.decoder_ffn_progressive or self.args.encoder_ffn_progressive:
            ffn_progressive_dimentions = list(map(int, self.args.progressive_ffn_dimentions.split(',')))
            ffn_progressive_num = len(ffn_progressive_dimentions)
            ffn_progressive_intervals = list(map(int, self.args.progressive_train_iters_ffn.split(',')))
            cnt = 0
            ffn_progressive_updates = []
            for interval in ffn_progressive_intervals:
                cnt += interval
                ffn_progressive_updates.append(cnt)
            assert len(ffn_progressive_updates) == len(ffn_progressive_dimentions)
            ffn_progressive_dimentions.append(self.config.d_ff)
            ffn_progressive_updates.append(self.args.train_iters)
            max_ffn_progressive_updates = ffn_progressive_updates[-2]
            ffn_progressive_index = 0
            logger.info(f"Progressive learning on FFN: increase dimensions after {str(ffn_progressive_updates)} updates.")
            # progressive_interval = 1
        # progressive train iters
        if not self.args.encoder_progressive_learning and not self.args.decoder_progressive_learning:
            max_progressive_updates = 0
        if not self.args.encoder_ffn_progressive and not self.args.decoder_ffn_progressive:
            max_ffn_progressive_updates = 0
        # setup optimizer
        optimizer_grouped_parameters, to_update_parameters, buffer_size = self.get_optimzied_group()
        logger.info("Optimized parameters number is {}".format(buffer_size))
        if self.data_parallel_size > 1 and self.args.tune_method == 'prompt':
            self.buffer = to_update_parameters[0].new(buffer_size)
            # print("buffer", self.args.rank, self.buffer)
        # print(optimizer_grouped_parameters)
        self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
        warm_up_steps = int(self.args.train_iters) * self.args.lr_warmup_fraction
        self.scheduler = get_scheduler(self.args.lr_decay_style, self.optimizer, warm_up_steps, self.args.train_iters)
        num_updates = 0
        log_dict = OrderedDict()
        best_metric = 0
        best_metric_dict = None
        best_num_updates = 0
        early_stop = 0
        begin_epoch = 0
        if self.args.load_prompt and not self.args.no_load_status:
            logger.info("Continue training from checkpoint({}) ...".format(self.args.load_prompt))
            begin_epoch, num_updates = self.load_train_status(self.args.load_prompt, self.args,
                                                              train_dataloader, train_sampler)
            # increase decoder or encoder
            if num_updates <= max_progressive_updates:
                if self.args.decoder_progressive_learning or self.args.encoder_progressive_learning:
                    while num_updates >= progressive_updates[progressive_index]:
                        progressive_index += 1
                    if progressive_index > 0:
                        new_layer_num = progressive_layers[progressive_index]
                        self.increase_model(self.args, self._model, num_updates, new_layer_num)
            # increase ffn dimentions
            if num_updates <= max_ffn_progressive_updates:
                if self.args.encoder_ffn_progressive or self.args.decoder_ffn_progressive:
                    while num_updates >= ffn_progressive_updates[ffn_progressive_index]:
                        ffn_progressive_index += 1
                    if ffn_progressive_index > 0:
                        new_ffn_dimension = ffn_progressive_dimentions[ffn_progressive_index]
                        self.modify_ffn_layers(self.args, self._model, new_ffn_dimension)
                        logger.info(f"num updates {num_updates}: Increase FFN layer dimension to {new_ffn_dimension}")
            # self.valid(epoch, num_updates)
            best_num_updates = num_updates
            if best_metric_dict is not None:
                best_metric = np.mean(best_metric_dict[key] for key in best_metric_dict)
                logger.info(f"Best averate score = {best_metric:.5f} at {best_num_updates}.")
        logger.info(f"Train {len(train_dataloader) // self.gradient_accumulation_steps} steps every epoch")
        logger.info(f" Gradient accumulation steps = {self.gradient_accumulation_steps}")
        for epoch in range(begin_epoch, self.args.train_epochs):
            self.optimizer.zero_grad()
            self.reset_logging(log_dict)
            if self.data_parallel_size > 1:
                # in distributed mode, calling the set_epoch() method at the beginning 
                # of each epoch before creating the DataLoader iterator is necessary to
                # make shuffling work properly across multiple epochs. Otherwise, 
                # the same ordering will be always used.
                train_sampler.set_epoch(epoch)
            for local_step, batch in enumerate(train_dataloader):
                # if local_step == 0:
                # torch.distributed.barrier()
                # temp = torch.masked_fill(batch['labels'][0], batch['labels'][0] == -100, 0)
                # print(self.tokenizer.convert_ids_to_tokens(temp))
                # print(self.tokenizer.decode(temp))
                # print(self.tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
                # torch.distributed.barrier()
                # quit()
                loss = self.train_step(batch)
                self.add_logging(log_dict, 'loss', loss.item() * self.gradient_accumulation_steps)
                if (local_step + 1) % self.gradient_accumulation_steps == 0:
                    # update model parameter
                    updated, old_scale = self.optimizer_step(to_update_parameters)
                    # print("{}: rank {} {}".format(num_updates, self.args.rank, str(self.prompt.grad)))
                    if updated:
                        num_updates += 1
                    else:
                        logger.info("Inf or NaN detected in grad. Change scale from {:.1f} to {:.1f}"\
                                    .format(old_scale, self.scaler.get_scale()))
                    if num_updates % self.args.log_interval == 0:
                        # to log
                        self.log_step(log_dict, tensorboard_suffix='train', epoch=epoch, num_updates=num_updates,
                                      lr=self.scheduler.get_last_lr()[0])
                    self.reset_logging(log_dict)
                    if self.args.valid_interval is not None and \
                            num_updates % self.args.valid_interval == 0:
                        current_metrics = self.valid(epoch, num_updates)
                        best_update, average_score = self.early_stop(current_metrics, best_metric, epoch, num_updates)
                        if not best_update:
                            early_stop += 1
                            logger.info(f"Early stop + 1 = {early_stop}. " \
                                        f"Best averate score = {best_metric:.5f} at {best_num_updates}.")
                        else:
                            early_stop = 0
                            best_metric = average_score
                            best_num_updates = num_updates
                        if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                            break
                    if self.args.save_interval is not None and \
                            num_updates % self.args.save_interval == 0:
                        save_path = f"{self.args.save}/checkpoint@{epoch}-{num_updates}.pt"
                        if self.rank == 0:
                            self.save_checkpoint(save_path, epoch, num_updates)
                        if self.data_parallel_size > 1:
                            dist.barrier()
                    # increase decoder or encoder
                    if num_updates <= max_progressive_updates:
                        if self.args.decoder_progressive_learning or self.args.encoder_progressive_learning:
                            if num_updates == progressive_updates[progressive_index]:
                                progressive_index += 1
                                new_layer_num = progressive_layers[progressive_index]
                                self.increase_model(self.args, self._model, num_updates, new_layer_num)
                    # increase ffn dimentions
                    if num_updates <= max_ffn_progressive_updates:
                        if self.args.encoder_ffn_progressive or self.args.decoder_ffn_progressive:
                            if num_updates == ffn_progressive_updates[ffn_progressive_index]:
                                ffn_progressive_index += 1
                                new_ffn_dimension = ffn_progressive_dimentions[ffn_progressive_index]
                                self.modify_ffn_layers(self.args, self._model, new_ffn_dimension)
                                logger.info(f"num updates {num_updates}: "
                                            f"Increase FFN layer dimension to {new_ffn_dimension}")
                    if num_updates >= self.args.train_iters:
                        break
            # early stop training
            # if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
            #     logger.info(f"Stop traning. Best averate score = {best_metric:.3f} at {best_num_updates}.")
            #     break
            # end of epoch, do valid
            # logger.info(f"End of epoch {epoch} ...")
            # current_metrics = self.valid(epoch, num_updates)
            # check whether to need early stop
            # best_update, average_score = self.early_stop(current_metrics, best_metric, epoch, num_updates)
            # if not best_update:
            #     early_stop += 1
            #     logger.info(f"Early stop + 1 = {early_stop}. "\
            #                 f"Best averate score = {best_metric:.3f} at {best_num_updates}.")
            # else:
            #     early_stop = 0
            #     best_metric = average_score
            #     best_num_updates = num_updates
            #     best_metric_dict = current_metrics
            if num_updates >= self.args.train_iters:
                break
            # save when epoch ends
            if self.args.save_every_epoch:
                save_path = f"{self.args.save}/checkpoint-last.pt"
                if self.rank == 0:
                    self.save_checkpoint(save_path, epoch, num_updates)
                if self.data_parallel_size > 1:
                    dist.barrier()
            if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                logger.info(f"Stop training. Best averate score = {best_metric:.5f} at {best_num_updates}.")
                break

    def early_stop(self, metrics, best_metric, epoch, num_updates):
        current_metric = 0
        update = True
        for key in metrics:
            current_metric += metrics[key]
        current_metric = current_metric / len(metrics)  # compare average
        if best_metric > current_metric:
            update = False
        else:
            save_path = f"{self.args.save}/checkpoint-best.pt"
            if self.rank == 0:
                self.save_checkpoint(save_path, epoch, num_updates)
            if self.data_parallel_size > 1:
                dist.barrier()
        return update, current_metric

    def valid(self, epoch=0, num_updates=0):
        # defines the validation process
        self.model.eval()
        valid_dataloader, valid_sampler = self.build_dataloader(self.args, self.valid_dataset,
                           batch_size=self.args.eval_micro_batch_size, shuffle=False)
        my_index = []
        my_prediction= []
        valid_log_dict = OrderedDict()
        logger.info("Begin validation on {:d} samples ...".format(len(self.valid_dataset)))
        metrics = {}
        with torch.no_grad():
            for local_step, batch in enumerate(valid_dataloader):
                # if local_step == 0:
                #     temp = torch.masked_fill(batch['labels'][1], batch['labels'][1] == -100, 0)
                #     print(self.tokenizer.convert_ids_to_tokens(temp))
                #     print(self.tokenizer.decode(temp))
                #     print(self.tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
                # quit()
                all_input = self.prepare_data(batch)
                output = self._model(**all_input)
                valid_loss = output.loss
                self.add_logging(valid_log_dict, 'loss', valid_loss.item())
                if self.args.task == 'language-model':
                    continue
                decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                generated_ids = self._model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.args.max_target_length,
                    early_stopping=True,
                    num_beams=self.args.beam_width,
                    length_penalty=self.args.length_penalty,
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                my_index.extend(batch['id'])
                my_prediction.extend(gen_text)
        if len(my_prediction) != 0:
            metrics = self.evaluate(my_prediction, my_index, self.valid_dataset)
        valid_loss = self.log_step(valid_log_dict, suffix="Valid |", tensorboard_suffix='valid',
                      epoch=epoch, num_updates=num_updates, **metrics)
        if not self.args.debug:
            self.model.train()
        # metrics['loss'] = - valid_loss
        return metrics

    def get_decoder_input_ids(self, inputs_embeds):
        decoder_start_token_id = self.config.decoder_start_token_id
        decoder_input_ids = (
                torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long, device=inputs_embeds.device) * decoder_start_token_id
        )
        return decoder_input_ids
    
    def evaluate(self, my_prediction, my_index, dataset):
        all_prediction = [None] * self.data_parallel_size
        all_index = [None] * self.data_parallel_size
        if self.data_parallel_size > 1:
            dist.all_gather_object(obj=my_prediction, object_list=all_prediction)
            dist.all_gather_object(obj=my_index, object_list=all_index)
        else:
            all_prediction = [my_prediction]
            all_index = [my_index]
        metrics = {}
        if self.rank == 0:
            cnt = 0
            predictions = []
            index = []
            for i in range(len(all_prediction[0])):
                for j in range(len(all_prediction)):
                    predictions.append(all_prediction[j][i])
                    index.append(all_index[j][i])
            predictions = predictions[:len(dataset)]
            index = index[:len(dataset)]
            metrics = dataset.evaluate(predictions)
        if self.data_parallel_size > 1:
            object_list = [metrics]
            dist.broadcast_object_list(object_list=object_list, src=0)
            metrics = object_list[0]
        return metrics
    
    def save_checkpoint(self, path, epoch, num_updates):
        state_dict = OrderedDict()
        if self.args.tune_method == 'model':
            # save model
            state_dict['model'] = self._model.state_dict()
        elif self.args.tune_method == 'prompt':
            # save prompt
            state_dict['prompt'] = self.prompt
            if self.args.add_cross_ffn:
                state_dict['cross_ffn'] = self._model.save_cross_ffn()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict['config'] = self.config
        state_dict['args'] = vars(self.args)
        state_dict['current_state'] = {'epoch': epoch, 'num_updates': num_updates}
        torch.save(state_dict, path)
        logger.info(f"epoch: {epoch} num_updates: {num_updates} Save {self.args.tune_method} to {path}.")
    
    def load_checkpoint(self, path, do_eval=False):
        state_dict = torch.load(path, map_location=self.device)
        if state_dict['args']['tune_method'] == 'model':
            # load model
            self._model.load_state_dict(state_dict['model'])
        elif state_dict['args']['tune_method'] == 'prompt':
            # load prompt
            self.prompt = state_dict['prompt']
            if self.args.add_cross_ffn:
                self._model.load_cross_ffn(state_dict['cross_ffn'])
        current_state = state_dict['current_state']
        logger.info(f"Load {state_dict['args']['tune_method']} from {path}.")
        if do_eval:
            logger.info(f"Evaluate on loaded {state_dict['args']['tune_method']} ...")
            metrics = self.valid(0, 0)
        return current_state

    def load_train_status(self, path, args, dataloader, sampler):
        state_dict = torch.load(path, map_location='cpu')
        if not args.no_load_optim:
            if 'optimizer' not in state_dict:
                logger.warning("Optimizer status not in checkpoint.")
            else:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                self.optimizer.load_state_dict(state_dict['optimizer'])
                logger.info("Load optimizer status from checkpoint.")
        current_state = state_dict['current_state']
        epoch, num_updates = current_state['epoch'], current_state['num_updates']
        steps_per_epoch = len(dataloader) // self.gradient_accumulation_steps
        current_step = num_updates % steps_per_epoch
        if current_step == 0:
            epoch += 1
        for e in range(epoch):
            if self.data_parallel_size > 1:
                sampler.set_epoch(e)
        # for local_step, _ in enumerate(dataloader):
        #     if local_step == current_step * self.gradient_accumulation_steps:
        #         break
        logger.info(f"Set train epoch {epoch}, num_updates {num_updates}")
        return epoch, num_updates


    def build_dataloader(self, args, dataset, batch_size, shuffle=True):
        sampler = None
        if self.data_parallel_size > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle, rank=self.rank)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                     shuffle=shuffle if sampler is None else None,
                     num_workers=self.args.num_workers, sampler=sampler,
                     collate_fn=dataset.collator, pin_memory=False)
        return dataloader, sampler

    def build_optimizer(self, args, params):
        optimizer = Adafactor(params, lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
        # optimizer = AdamW(params, lr=args.lr)
        return optimizer

    def prepare_model_data(self, batch):
        all_input = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'labels': batch['labels'].to(self.device),
            'decoder_attention_mask': batch['decoder_attention_mask'].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self._model.get_input_embeddings()(input_ids)
        all_input['inputs_embeds'] = input_embeds
        return all_input

    def prepare_prompt_data(self, batch):
        all_input = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'labels': batch['labels'].to(self.device),
            'decoder_attention_mask': batch['decoder_attention_mask'].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self._model.get_input_embeddings()(input_ids)
        batch_size = input_ids.shape[0]
        prompt = torch.unsqueeze(self.prompt, dim=0).expand((batch_size,) + self.prompt.shape)
        prompt_attention = torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device)
        # cat prompt with input ids
        input_embeds = torch.cat((prompt, input_embeds), dim=1)
        # cat prompt attention mask to initial attention mask
        all_input['attention_mask'] = torch.cat((prompt_attention, all_input['attention_mask']), dim=1)
        # print("input_embeds", input_embeds.shape)
        all_input['inputs_embeds'] = input_embeds
        return all_input
    
    def train_step(self, batch):
        with autocast(enabled=self.args.fp16):
            all_input = self.prepare_data(batch)
            output = self.model(**all_input)
            loss = output.loss / self.gradient_accumulation_steps
        if self.args.fp16:
            # Scales the loss, and calls backward()
            # to create scaled gradients
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss
    
    def optimizer_step(self, parameters):
        updated = True
        scale = 0
        if self.data_parallel_size > 1 and self.args.tune_method == 'prompt':
            self.reduce_grads(parameters)
        if self.args.fp16:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, self.args.max_norm)
            self.scaler.step(self.optimizer)
            scale = self.scaler.get_scale()
            self.scaler.update()
            updated = (scale == self.scaler.get_scale())
        else:
            torch.nn.utils.clip_grad_norm_(parameters, self.args.max_norm)
            self.optimizer.step()
        if updated:
            self.scheduler.step()
        self.optimizer.zero_grad()
        return updated, scale
    
    def log_step(self, log_dict, suffix='', tensorboard_suffix=None, **kwargs):
        new_log_dict = OrderedDict()
        for key, value in kwargs.items():
            new_log_dict[key] = value
        for key in log_dict:
            key_tensor = torch.tensor(log_dict[key], device=self.device)
            if self.data_parallel_size > 1:
                dist.all_reduce(key_tensor, op=dist.ReduceOp.SUM)
            key_value = (key_tensor / self.data_parallel_size).mean().item()
            new_log_dict[key] = key_value
        message = '' + suffix
        if 'loss' in new_log_dict and self.args.task == 'language-model':
            new_log_dict['ppl'] = 2 ** new_log_dict['loss']
        for key, value in new_log_dict.items():
            if isinstance(value, float):
                message += ' {:s}: {:.5f}'.format(key, value)
            else:
                message += ' {:s}: {}'.format(key, value)
        logger.info(message)
        if self.tensorboard is not None:
            for key, value in new_log_dict.items():
                if key in ['epoch', 'num_updates']:
                    continue
                tag = f'{tensorboard_suffix}/{key}' if tensorboard_suffix is not None else key
                global_step = kwargs.get('num_updates', None)
                self.tensorboard.add_scalar(tag, value, global_step=global_step)
        return new_log_dict.get('loss', None)
    
    def add_logging(self, log_dict, key, value):
        if key not in log_dict:
            log_dict[key] = []
        log_dict[key].append(value)
    
    def reset_logging(self, log_dict):
        for key in log_dict:
            log_dict[key] = []

    def prune_ffn_layer(self, d_ff, current_layer, layer_idx,
                        new_config, select_method, full_ff_layer = None):
        # do FFN reduction on one layer
        keep_num = new_config.d_ff
        device = current_layer.layer[-1].DenseReluDense.wo.weight.data.device
        ff_layer = current_layer.layer[-1].DenseReluDense.cpu()
        if full_ff_layer is None:
            full_ff_layer = ff_layer
        new_fflayer = T5DenseGatedGeluDense(new_config)
        score_layer_idx = layer_idx
        if new_config.is_decoder:
            score_layer_idx += self.config.num_layers
        keep_index = NEURON_METHOD[select_method](keep_num, d_ff, scores=self.scores, layer_idx=score_layer_idx)
        new_fflayer.wi_0.weight.data = full_ff_layer.wi_0.weight.data[keep_index, :]
        new_fflayer.wi_1.weight.data = full_ff_layer.wi_1.weight.data[keep_index, :]
        new_fflayer.wo.weight.data = full_ff_layer.wo.weight.data[:, keep_index]
        new_fflayer.to(device)
        for n, p in new_fflayer.named_parameters():
            p.requires_grad = False
        new_fflayer.train()
        current_layer.layer[-1].DenseReluDense = new_fflayer
        return full_ff_layer

    def modify_ffn_layers(self, args, model, new_ffn_dimension):
        # Do FFN Reduction. new_ffn_dimentsion defines the number of new dimensions in each layer
        if args.encoder_ffn_progressive:
            new_config = copy.deepcopy(self.config)
            new_config.d_ff = new_ffn_dimension
            encoder_full_layers = self.full_layers['encoder']
            for idx in range(self.current_config.num_layers):
                layer_idx = self.layers_index['encoder'][idx]
                full_ff_layer = self.prune_ffn_layer(self.config.d_ff, model.encoder.block[idx], layer_idx,
                                                     new_config, args.select_ffn_method, encoder_full_layers[layer_idx])
                encoder_full_layers[layer_idx] = full_ff_layer
                # logger.info("encoder {} dropout: {}".format(layer_idx, new_config.dropout_rate))
        if args.decoder_ffn_progressive:
            new_config = copy.deepcopy(self.config)
            new_config.d_ff = new_ffn_dimension
            new_config.is_decoder = True
            decoder_full_layers = self.full_layers['decoder']
            for idx in range(self.current_config.num_decoder_layers):
                layer_idx = self.layers_index['decoder'][idx]
                full_ff_layer = self.prune_ffn_layer(self.config.d_ff, model.decoder.block[idx], layer_idx,
                                                     new_config, args.select_ffn_method, decoder_full_layers[layer_idx])
                decoder_full_layers[layer_idx] = full_ff_layer
                # logger.info("decoder {} dropout {}".format(layer_idx, new_config.dropout_rate))
        self.current_config.d_ff = new_ffn_dimension
        self.wrape_model = None

    def prune_decoder_layers(self, args, layers, model, full_decoder=None):
        # Layer Dropping in Decoder layers.
        num_layers = len(layers)
        from t5.modeling_t5 import T5Stack
        decoder_config = copy.deepcopy(model.config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = num_layers
        if full_decoder is None:
            full_decoder = model.decoder  # store full decoder
            full_decoder = full_decoder.cpu()  # move to cpu
        new_decoder = T5Stack(decoder_config, model.shared)
        for index, layer_index in enumerate(layers):
            new_decoder.block[index] = copy.deepcopy(full_decoder.block[layer_index])
        new_decoder.final_layer_norm = copy.deepcopy(full_decoder.final_layer_norm)
        model.decoder = new_decoder
        return full_decoder

    def prune_encoder_layers(self, args, layers, model, full_encoder=None):
        # Layer Dropping in Encoder layers.
        num_layers = len(layers)
        from t5.modeling_t5 import T5Stack
        encoder_config = copy.deepcopy(model.config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.num_layers = num_layers
        if full_encoder is None:
            full_encoder = model.encoder  # store full decoder
            full_encoder = full_encoder.cpu()  # move to cpu
        new_encoder = T5Stack(encoder_config, model.shared)
        for index, layer_index in enumerate(layers):
            new_encoder.block[index] = copy.deepcopy(full_encoder.block[layer_index])
        new_encoder.final_layer_norm = copy.deepcopy(full_encoder.final_layer_norm)
        model.encoder = new_encoder
        return full_encoder

    def increase_model(self, args, model, num_updates, new_layer_num):
        if args.encoder_progressive_learning:
            num_layers = self.config.num_layers
            if len(model.encoder.block) != len(self.full_encoder.block):
                layers = SELECT_METHOD[args.select_layer_method](new_layer_num, num_layers, scores=self.layer_scores)
                model.encoder.cpu()
                init_layer_num = len(model.encoder.block)
                self.prune_encoder_layers(args, layers, model, self.full_encoder)
                if args.model_parallel_size > 1:
                    self.current_device_maps['encoder'] = self.modify_device_maps(self.device_maps, layers)
                    model.encoder.parallelize(self.current_device_maps['encoder'])
                else:
                    model.encoder.to(self.device)
                model.encoder.train()
                for n, p in model.encoder.named_parameters():
                    p.requires_grad = False
                # logger.info("increase encoder {}".format(model.encoder.config.dropout_rate))
                self.current_config.num_layers = new_layer_num
                self.layers_index['encoder'] = layers
                logger.info(f"num updates {num_updates}: "
                            f"incease encoder layer number from {init_layer_num} to {len(model.encoder.block)}. "
                            f"new layer distribution is {layers}")
        if args.decoder_progressive_learning:
            num_layers = self.config.num_decoder_layers
            if len(model.decoder.block) != len(self.full_decoder.block):
                layers = SELECT_METHOD[args.select_layer_method](new_layer_num, num_layers,
                                                                 scores=self.layer_scores,
                                                                 num_encoder_layers=self.config.num_layers)
                model.decoder.cpu()
                init_layer_num = len(model.decoder.block)
                self.prune_decoder_layers(args, layers, model, self.full_decoder)
                if args.model_parallel_size > 1:
                    self.current_device_maps['decoder'] = self.modify_device_maps(self.device_maps, layers)
                    model.decoder.parallelize(self.current_device_maps['decoder'])
                else:
                    model.decoder.to(self.device)
                model.decoder.train()
                for n, p in model.decoder.named_parameters():
                    p.requires_grad = False
                self.current_config.num_decoder_layers = new_layer_num
                self.layers_index['decoder'] = layers
                # logger.info("increase decoder".format(model.decoder.config.dropout_rate))
                logger.info(f"num updates {num_updates}: "
                            f"incease decoder layer number from {init_layer_num} to {len(model.decoder.block)}. "
                            f"new layer distribution is {layers}")
        if self.current_config.d_ff != self.config.d_ff:
            self.modify_ffn_layers(args, model, self.current_config.d_ff)
        if self.args.model_parallel_size > 1:
            logger.info(str(self.current_device_maps))
        torch.cuda.empty_cache()
        self.wrape_model = None

    def reduce_grads(self, parameters):
        offset = 0
        buffer = self.buffer
        for p in parameters:
            sz = p.numel()
            if p.grad is not None:
                buffer[offset: offset + sz].copy_(p.grad.data.view(-1))
            else:
                buffer[offset: offset + sz].zero_()
            offset += sz
        # print(buffer)
        dist.all_reduce(buffer)
        buffer = buffer / self.args.data_parallel_size
        # copy all-reduced grads back into their original place
        offset = 0
        for p in parameters:
            sz = p.numel()
            if p.grad is not None:
                p.grad.data.copy_(buffer[offset: offset + sz].view_as(p))
            else:
                p.grad = buffer[offset: offset + sz].view_as(p).clone()
            offset += sz

    def set_dataset_max_target_length(self, datasets, max_target_length):
        for d in datasets:
            if d is not None:
                d.set_max_target_length(max_target_length)

    def modify_device_maps(self, original_maps, new_layers):
        new_maps = {}
        for key in original_maps:
            new_maps[key] = []
        for index, layer_index in enumerate(new_layers):
            for key in original_maps:
                if layer_index in original_maps[key]:
                    new_maps[key].append(index)
                    break
        for key in new_maps:
            if len(new_maps[key]) == 0:
                new_maps.pop(key)
        return new_maps

