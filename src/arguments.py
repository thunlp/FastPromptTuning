# This file defines the basic arguments of this project
# Author: Yufei Huang
# Date: 2022-10-17
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import argparse
import os

# MODEL_CHOICES=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']


def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Arguments',
                                     allow_abbrev=False)
    parser = _add_network_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_prompt_args(parser)
    parser = _add_train_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    if args.config_name_or_path is None:
        args.config_name_or_path = args.model_name_or_path

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1')) * args.model_parallel_size
    assert args.world_size % args.model_parallel_size == 0, \
        f"World size {args.world_size} should be multiple of model parallel size {args.model_parallel_size}"
    args.data_parallel_size = args.world_size // args.model_parallel_size
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, model-parallel-size: {}'.format(
                  args.world_size, args.data_parallel_size, args.model_parallel_size), flush=True)
    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])
    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.eval_micro_batch_size is None:
        args.eval_micro_batch_size = args.micro_batch_size
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    assert (args.train_epochs is not None) or (args.train_iters is not None) or (not args.train)
    if args.tune_method == "model":
        args.prompt_length = 0
        if args.rank == 0:
            print('setting prompt length to 0, since fine-tuning donesn\'t need prompt.')
    if args.load_prompt is not None and not os.path.isfile(args.load_prompt):
        args.load_prompt = None
        if args.rank == 0:
            print('load prompt path is not a file, setting load path to None.')
    _print_args(args)

    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)


def _add_network_args(parser):
    group = parser.add_argument_group(title='network config')

    group.add_argument("--model-name-or-path", type=str, default="t5-large",
                        help="pretrained model name or choice.")
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                        help="tokenizer name or path.")
    group.add_argument("--config-name-or-path", type=str, default=None,
                        help="config name or path.")
    group.add_argument("--cache-dir", type=str, default=None,
                        help="cache dir for downloading model from huggingface.")
    group.add_argument("--model-parallel-size", type=int, default=1,
                        help="Model parallel size.")
    group.add_argument("--beam-width", type=int, default=1, help="Beam search size for generation.")
    group.add_argument("--length-penalty", type=float, default=1.0, help="Leagth penalty for generation.")

    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-interval', type=int, default=1,
                       help='Report interval.')
    group.add_argument('--tensorboard-dir', type=str, default=None)

    return parser


def _add_prompt_args(parser):
    group = parser.add_argument_group(title='prompt')

    group.add_argument('--prompt-length', type=int, default=0,
                       help='The length of prompt.')
    group.add_argument('--prompt-init', type=str, choices=['uniform', 'embedding'], default='uniform',
                       help='Initialization method for prompt tuning.')
    group.add_argument('--overwrite-prompt-length', action='store_true',
                       help="overwrite prompt length in config.")
    # prune ffn layer
    group.add_argument('--prune-neuron', action='store_true',
                       help="Whether to prune knowledge neurons in model.")
    group.add_argument("--score-file", type=str, default=None,
                       help="file to load scores for each neuron")
    group.add_argument("--score-file-layer", type=str, default=None,
                       help="file to load scores for each neuron")
    group.add_argument("--prune-rate", type=float, default=None,
                       help="rate for prunning neurons numbers.")
    # early exit
    group.add_argument("--early-exit", action='store_true', help="early exit in decoder.")
    group.add_argument("--decoder-keep-layers", default='all', type=str,
                       help="the index of layers keeped in decoder, should used when early-exit is true.")
    group.add_argument("--encoder-keep-layers", default='all', type=str,
                       help="the index of layers keeped in encoder, should used when early-exit is true."
                            " all represent all layers is kept.")
    # ealy backpropagation in layers
    group.add_argument("--decoder-progressive-learning", action="store_true",
                       help='to increase decoder layer num gradually in training.')
    group.add_argument("--encoder-progressive-learning", action="store_true",
                       help='to increase decoder layer num gradually in training.')
    group.add_argument("--progressive-layers", type=str, default=None,
                       help="which layers to use in progressive learning, split by comma. e.g 3,6,12")
    group.add_argument("--select-layer-method", type=str, default='sequence', choices=['sequence', 'evenly', 'score'],
                       help="how to select layers for progressive learning.")
    group.add_argument("--progressive-train-iters", type=str, default=None,
                       help="number of train iterations for pregressive learning.")
    group.add_argument("--add-cross-ffn", action='store_true', help="add ffn layer between encoder and decoder.")
    # ffn layer progressive learning
    group.add_argument("--encoder-ffn-progressive", action="store_true",
                       help="encoder ffn layer progressive learning.")
    group.add_argument("--decoder-ffn-progressive", action="store_true",
                       help="decoder ffn layer progressive learning.")
    group.add_argument("--progressive-ffn-dimentions", type=str, default=None,
                       help="how many dimensions used for pregressive learning in ffn layer, split by \",\".")
    group.add_argument("--progressive-train-iters-ffn", type=str, default=None,
                       help="number of train iterations for pregressive learning in ffn layer.")
    group.add_argument("--select-ffn-method", type=str, default='sequence', choices=['sequence', 'score'],
                       help="how to select ffn parameters for progressive learning.")
    group.add_argument("--debug", action="store_true",
                       help="open dropout when increase model size.")
    return parser


def _add_train_args(parser):
    group = parser.add_argument_group(title='train')

    group.add_argument('--train', action="store_true")
    group.add_argument('--test', action="store_true")
    group.add_argument('--task', type=str, default="multirc",
                       help='task name.')
    group.add_argument('--datapath', type=str, default="superglue/MultiRC")
    group.add_argument('--max-seq-length', type=int, default=512)
    group.add_argument('--max-target-length', type=int, default=64)
    group.add_argument("--dropout", type=float, default=None)
    group.add_argument('--no-cuda', action="store_true")
    group.add_argument('--num-workers', type=int, default=4,
                       help='num workers for dataloader.')
    group.add_argument('--fp16', action='store_true')
    group.add_argument('--seed', type=int, default=None)
    group.add_argument('--max-norm', type=float, default=1.0)
    group.add_argument('--early-stop', type=int, default=-1)
    group.add_argument('--tune-method', type=str, default="prompt",
                       choices=['prompt', "model"], help='Report interval.')
    group.add_argument('--train-epochs', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-iters', type=int, default=None)
    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--eval-micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                            'Global batch size is local batch size times data '
                            'parallel size times number of micro batches.')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--valid-interval', type=int, default=None,
                       help='valid every interval.')
    group.add_argument("--local_rank", type=int, default=-1)
    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--weight-decay', type=float, default=1.0)
    group.add_argument('--cross-ffn-lr', type=float, default=None)

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--save-every-epoch', action='store_true',
                       help="Save checkpoint when epoch ends.")
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--load-model', type=str,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--load-prompt', type=str,
                       help='Directory containing a prompt checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-status', action='store_true',
                       help='Do not load train status when loading checkpoint.')
    return parser
