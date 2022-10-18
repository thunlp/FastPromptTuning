# This file comes from https://github.com/EleutherAI/knowledge-neurons
# Copyright (c) 2021 Sid Black

# main knowledge neurons class
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import collections
from typing import List, Optional, Tuple, Callable
import math
from functools import partial
from transformers import PreTrainedTokenizerBase
from .patch import *


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None

        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif "t5" in model_type:
            self.encoder_transformer_layers_attr = 'encoder.block'
            self.decoder_transformer_layers_attr = 'decoder.block'
            self.input_ff_attr = 'DenseReluDense'
            self.output_ff_attr = 'DenseReluDense.wo.weight'
            self.word_embeddings_attr = 'shared'
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self, attr_name):
        return get_attributes(self.model, attr_name)

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        else:
            # with autoregressive models we always want to target the last token
            mask_idx = -1
        if target is not None:
            if "gpt" in self.model_type:
                target = self.tokenizer.encode(target)
            else:
                target = self.tokenizer.convert_tokens_to_ids(target)
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        if self.model_type == "gpt":
            n_sampling_steps = len(target_label)
        else:
            n_sampling_steps = 1  # TODO: we might want to use multiple mask tokens even with bert models

        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            if n_sampling_steps > 1:
                target_idx = target_label[i]
            else:
                target_idx = target_label
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            all_argmax_probs.append(argmax_prob)

            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self, attr_name):
        return len(self._get_transformer_layers(attr_name))

    def intermediate_size(self):
        return self.model.config.d_ff

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = activations.repeat((steps, 1))
        out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
        )
        return out

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int, transformer_layers_attr: str,
        attention_mask: torch.Tensor = None
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                if mask_idx is not None:
                    self.baseline_activations = torch.clone(acts[:, mask_idx, :]).detach()
                else:
                    assert attention_mask is not None
                    self.baseline_activations = torch.clone(acts[attention_mask != 0]).detach()

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_baseline_with_activations_for_all_layers(
        self, encoded_input: dict, attention_mask: torch.Tensor, decoder_attention_mask: torch.Tensor,
    ):
        def get_activations(model, layer_idx, transformer_layers_attr, attention):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                baseline_activations = torch.clone(acts[attention != 0]).detach()
                if self.baseline_activations is None:
                    self.baseline_activations = []
                self.baseline_activations.append(baseline_activations.to(self.device))
            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )
        handles = []
        # register hook
        for layer_idx in range(self.n_layers(self.encoder_transformer_layers_attr)):
            handles.append(
                get_activations(self.model, layer_idx=layer_idx, attention=attention_mask,
                                transformer_layers_attr=self.encoder_transformer_layers_attr)
            )
        for layer_idx in range(self.n_layers(self.decoder_transformer_layers_attr)):
            handles.append(
                get_activations(self.model, layer_idx=layer_idx, attention=decoder_attention_mask,
                                transformer_layers_attr=self.decoder_transformer_layers_attr)
            )
        baseline_outputs = self.model(**encoded_input)
        # remove hook
        for h in handles:
            h.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores(
        self,
        input_embeb: torch.Tensor,
        label: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_id: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        target_length: int = 10,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        scores = []
        if attribution_method != "integrated_grads":
            encoded_input = {
                "inputs_embeds": input_embeb,
                "attention_mask": attention_mask,
                "labels": label,
                "decoder_attention_mask": decoder_attention_mask,
            }
            baseline_outputs, baseline_activations = self.get_baseline_with_activations_for_all_layers(
                encoded_input,
                attention_mask,
                decoder_attention_mask
            )
            for index, activations in enumerate(baseline_activations):
                if attribution_method == 'abs_activations':
                    activations = torch.abs(activations)
                scores.append(activations.mean(dim=0))
        else:
            # can only get score layer by layer
            for layer_idx in tqdm(
                range(self.n_layers(self.encoder_transformer_layers_attr)),
                desc="Getting attribution scores for encoder layer...",
                disable=not pbar,
            ):
                layer_scores = self.get_scores_for_layer(
                    input_embeb,
                    attention_mask,
                    decoder_input_id,
                    decoder_attention_mask=decoder_attention_mask,
                    label=label,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.encoder_transformer_layers_attr,
                    target_length=target_length,
                    batch_size=batch_size,
                    steps=steps,
                    attribution_method=attribution_method,
                    mask_idx=None
                )
                scores.append(layer_scores)
            for layer_idx in tqdm(
                range(self.n_layers(self.decoder_transformer_layers_attr)),
                desc="Getting attribution scores for decoder layer...",
                disable=not pbar,
            ):
                layer_scores = self.get_scores_for_layer(
                    input_embeb,
                    attention_mask,
                    decoder_input_id,
                    decoder_attention_mask=decoder_attention_mask,
                    label=label,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.decoder_transformer_layers_attr,
                    target_length=target_length,
                    batch_size=batch_size,
                    steps=steps,
                    attribution_method=attribution_method,
                    mask_idx=None,
                )
                # input()
                scores.append(layer_scores)
        return torch.stack(scores)

    def get_coarse_neurons(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ) -> List[List[int]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            return torch.nonzero(attribution_scores > threshold).cpu().tolist()
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            return (
                torch.nonzero(attribution_scores > np.percentile(s, percentile))
                .cpu()
                .tolist()
            )

    def get_refined_neurons(
        self,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = 0.3,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        quiet=False,
    ) -> List[List[int]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.

        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        for prompt in tqdm(
            prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            coarse_neurons.append(
                self.get_coarse_neurons(
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                )
            )
        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                negative_examples,
                desc="Getting coarse neurons for negative examples",
                disable=quiet,
            ):
                negative_neurons.append(
                    self.get_coarse_neurons(
                        negative_example,
                        ground_truth,
                        batch_size=batch_size,
                        steps=steps,
                        adaptive_threshold=coarse_adaptive_threshold,
                        threshold=coarse_threshold,
                        percentile=coarse_percentile,
                        pbar=False,
                    )
                )
        if not quiet:
            total_coarse_neurons = sum([len(i) for i in coarse_neurons])
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        refined_neurons = []
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        for neuron, count in c.items():
            if count > t:
                refined_neurons.append(list(neuron))

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons

    def get_scores_for_layer(
        self,
        input_embed: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_id: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        label: torch.Tensor,
        layer_idx: int,
        transformer_layers_attr: str,
        target_length: int = 10,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        mask_idx=-1,
    ):
        """
        get the attribution scores for a given layer
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = target_length

        if attribution_method == "integrated_grads":
            integrated_grads = []

            for i in range(n_sampling_steps):
                encoded_input = {
                    "inputs_embeds": input_embed,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_id,
                }
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx, transformer_layers_attr
                )
                # Now we want to gradually change the intermediate activations of our
                # layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0,
                # the second by 0.1, etc., until we reach 1
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position,
                    # with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "inputs_embeds": encoded_input["inputs_embeds"].repeat((batch_size, 1, 1)),
                        "attention_mask": encoded_input["attention_mask"].repeat((batch_size, 1)),
                        "decoder_input_ids": encoded_input["decoder_input_ids"].repeat((batch_size, 1)),
                    }

                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    target_idx = label[0][i]
                    grad = torch.autograd.grad(
                        torch.unbind(probs[:, target_idx]), batch_weights
                    )[0]
                    grad = grad.sum(dim=0)
                    integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step)

                # early stop
                if i + 1 < n_sampling_steps and label[0][i + 1] >= 0:
                    # next_token_str = self.tokenizer.decode(label[0][i + 1])
                    decoder_input_id = torch.cat((decoder_input_id, label[0][i + 1].view(1, 1)), dim=1)
                else:
                    break
            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )
            return integrated_grads
        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                encoded_input = {
                    "inputs_embeds": input_embed,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_id,
                }
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx, transformer_layers_attr
                )
                activations.append(baseline_activations)
                # early stop
                if i + 1 < n_sampling_steps and label[0][i + 1] >= 0:
                    # next_token_str = self.tokenizer.decode(label[0][i + 1])
                    decoder_input_id = torch.cat((decoder_input_id, label[0][i + 1].view(1, 1)), dim=1)
                else:
                    break
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        elif attribution_method == "abs_activations":
            activations = []
            for i in range(n_sampling_steps):
                encoded_input = {
                    "inputs_embeds": input_embed,
                    "attention_mask": attention_mask,
                    "labels": label,
                    "decoder_attention_mask": decoder_attention_mask,
                }
                if 'decoder' in transformer_layers_attr:
                    (
                        baseline_outputs,
                        baseline_activations,
                    ) = self.get_baseline_with_activations(
                        encoded_input, layer_idx, mask_idx, transformer_layers_attr, decoder_attention_mask
                    )
                else:
                    (
                        baseline_outputs,
                        baseline_activations,
                    ) = self.get_baseline_with_activations(
                        encoded_input, layer_idx, mask_idx, transformer_layers_attr, attention_mask
                    )
                if 'decoder' in transformer_layers_attr:
                    print(baseline_outputs['logits'].shape)
                    # print(baseline_outputs['past_key_values'].shape)
                    print(baseline_outputs['encoder_last_hidden_state'].shape)
                    print(baseline_activations.shape)
                    quit()
                activations.append(torch.abs(baseline_activations))
                # if label is not None:
                #     # early stop
                #     if label[0][i + 1] < 0:
                #         break
                #     # next_token_str = self.tokenizer.decode(label[0][i + 1])
                #     decoder_input_id = torch.cat((decoder_input_id, label[0][i + 1].view(1, 1)), dim=1)
            activations = torch.cat(activations, dim=0)
            activations = activations.mean(dim=0)
            return activations.squeeze(0)
        else:
            raise NotImplementedError

    def modify_activations(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        mode: str = "suppress",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the groundtruth being generated + the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch so we can unpatch them later
        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def suppress_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
        and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="suppress",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def enhance_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    @torch.no_grad()
    def modify_weights(
        self,
        prompt: str,
        neurons: List[List[int]],
        target: str,
        mode: str = "edit",
        erase_value: str = "zero",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        Update the *weights* of the neural net in the positions specified by `neurons`.
        Specifically, the weights of the second Linear layer in the ff are updated by adding or subtracting the value
        of the word embeddings for `target`.
        """
        assert mode in ["edit", "erase"]
        assert erase_value in ["zero", "unk"]
        results_dict = {}

        _, _, target_label = self._prepare_inputs(prompt, target)
        # get the baseline probabilities of the target being generated + the argmax / greedy completion before modifying the weights
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # get the word embedding values of the baseline + target predictions
        word_embeddings_weights = self._get_word_embeddings()
        if mode == "edit":
            assert (
                self.model_type == "bert"
            ), "edit mode currently only working for bert models - TODO"
            original_prediction_id = argmax_tokens[0]
            original_prediction_embedding = word_embeddings_weights[
                original_prediction_id
            ]
            target_embedding = word_embeddings_weights[target_label]

        if erase_value == "zero":
            erase_value = 0
        else:
            assert self.model_type == "bert", "GPT models don't have an unk token"
            erase_value = word_embeddings_weights[self.unk_token]

        # modify the weights by subtracting the original prediction's word embedding
        # and adding the target embedding
        original_weight_values = []  # to reverse the action later
        for layer_idx, position in neurons:
            output_ff_weights = self._get_output_ff_layer(layer_idx)
            if self.model_type == "gpt2":
                # since gpt2 uses a conv1d layer instead of a linear layer in the ff block, the weights are in a different format
                original_weight_values.append(
                    output_ff_weights[position, :].detach().clone()
                )
            else:
                original_weight_values.append(
                    output_ff_weights[:, position].detach().clone()
                )
            if mode == "edit":
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] -= original_prediction_embedding * 2
                    output_ff_weights[position, :] += target_embedding * 2
                else:
                    output_ff_weights[:, position] -= original_prediction_embedding * 2
                    output_ff_weights[:, position] += target_embedding * 2
            else:
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] = erase_value
                else:
                    output_ff_weights[:, position] = erase_value

        # get the probabilities of the target being generated + the argmax / greedy completion after modifying the weights
        (
            new_gt_prob,
            new_argmax_prob,
            new_argmax_completion_str,
            new_argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        def unpatch_fn():
            # reverse modified weights
            for idx, (layer_idx, position) in enumerate(neurons):
                output_ff_weights = self._get_output_ff_layer(layer_idx)
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] = original_weight_values[idx]
                else:
                    output_ff_weights[:, position] = original_weight_values[idx]

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def edit_knowledge(
        self,
        prompt: str,
        target: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="edit",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def erase_knowledge(
        self,
        prompt: str,
        neurons: List[List[int]],
        erase_value: str = "zero",
        target: Optional[str] = None,
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="erase",
            erase_value=erase_value,
            undo_modification=undo_modification,
            quiet=quiet,
        )
