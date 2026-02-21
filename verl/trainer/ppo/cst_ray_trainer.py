# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Modified Prompt PPO Trainer that extends RayPPOTrainer with prompt modification functionality.
"""

import copy
import re
from typing import Any, Callable, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import numpy as np
import torch

from verl import DataProto
from verl.utils.debug import marked_timer
from .ray_trainer import RayPPOTrainer
from collections import defaultdict
from verl.trainer.ppo.metric_utils import process_validation_metrics
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation import extract_distillation_inputs
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_distillation_policy,
    need_reference_policy,
    need_reward_model,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.stages import Stage
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl.trainer.ppo.ray_trainer import *


class CSTRayPPOTrainer(RayPPOTrainer):
    """
    PPO trainer that allows modifying prompts for reference policy computation.
    
    This trainer extends the base RayPPOTrainer to support different prompt modifications
    when computing reference log probabilities, enabling experiments with different
    prompt formulations for the reference policy.
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
        prompt_modifier: Optional[Callable[[str], str]] = None,
        prompt_modifier_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer with prompt modification capability.
        
        Args:
            prompt_modifier: Function that takes a prompt string and returns modified prompt string
            prompt_modifier_config: Configuration for prompt modification (e.g., templates, prefixes)
            ... (other args same as RayPPOTrainer)
        """
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        
        # Initialize prompt_modifier_config first
        self.prompt_modifier_config = prompt_modifier_config or {}
        
        # Set prompt_modifier - use provided one or default to 'keep_original'
        if prompt_modifier is not None:
            self.prompt_modifier = prompt_modifier
        else:
            # Get default modifiers and set to 'keep_original'
            default_modifiers = self._get_default_prompt_modifiers()
            self.prompt_modifier = default_modifiers["keep_original"]
    
    def _get_default_prompt_modifiers(self) -> Dict[str, Callable[[str], str]]:
        """
        Get a dictionary of common prompt modification functions.
        
        Returns:
            Dictionary mapping modifier names to modifier functions
        """
        def add_sft(prompt: str, extra_info=None) -> str:
            if extra_info is None or "answer" not in extra_info:
                return prompt
            return f"""{prompt} 
You may use the expert trajectory only as *private guidance* to check your own reasoning.
Do NOT quote, copy, paraphrase, or explicitly reference any sentence from it. 
Expert trajectory:{extra_info["answer"]}
Now solve the problem with your own step-by-step reasoning:"""
        
        def keep_original(prompt: str, extra_info=None) -> str:
            return prompt
        
        return {
            "add_sft": add_sft,
            "keep_original": keep_original,
        }
    
    def set_prompt_modifier(self, modifier_name: str, **kwargs):
        """
        Set prompt modifier by name using built-in modifiers.
        
        Args:
            modifier_name: Name of the modifier ("add_sft", "keep_original", etc.)
            **kwargs: Configuration for the modifier
        """
        self.prompt_modifier_config.update(kwargs)
        modifiers = self._get_default_prompt_modifiers()
        
        if modifier_name not in modifiers:
            raise ValueError(f"Unknown modifier: {modifier_name}. Available: {list(modifiers.keys())}")
        
        self.prompt_modifier = modifiers[modifier_name]
    
    def _extract_chat_messages(self, chat_prompt: str, extra_info=None) -> list:
        """
        Extract system and user messages from chat template format.
        
        Args:
            chat_prompt: Full chat template string
            extra_info: Additional information to pass to prompt modifier
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        messages = []
        
        # Extract system message
        system_pattern = r"<\|im_start\|>system\n(.*?)<\|im_end\|>"
        system_match = re.search(system_pattern, chat_prompt, re.DOTALL)
        if system_match:
            system_content = system_match.group(1).strip()
            messages.append({"role": "system", "content": system_content})
        
        # Extract user message
        user_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        user_match = re.search(user_pattern, chat_prompt, re.DOTALL)
        if user_match:
            user_content = user_match.group(1).strip()
            # Apply modification to user content with extra_info
            modified_user_content = self.prompt_modifier(user_content, extra_info)
            messages.append({"role": "user", "content": modified_user_content})
        
        return messages
    
    def _rebuild_chat_prompt(self, messages: list) -> str:
        """
        Rebuild chat prompt using tokenizer.apply_chat_template.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Rebuilt chat template string
        """
        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            # Fallback: just use user content if no chat template
            user_message = next((msg for msg in messages if msg["role"] == "user"), None)
            prompt_with_chat_template = user_message["content"] if user_message else ""
            
        return prompt_with_chat_template
    
    def _modify_prompts_for_ref_policy(self, batch: DataProto) -> DataProto:
        """
        Modify prompts in the batch for reference policy computation while preserving prompt+response structure.
        
        Args:
            batch: Original batch containing prompts and responses
            
        Returns:
            Modified batch with updated prompts combined with original responses
        """
        
        # Create a copy to avoid modifying the original batch
        modified_batch = copy.deepcopy(batch)

        prompt_ids = batch.batch["prompts"]
        response_ids = batch.batch["responses"]

        # Check if dataset already contains pre-computed modified prompts (as text)
        # breakpoint()
        if "modified_prompt_texts" in batch.non_tensor_batch:
            # Use pre-computed modified prompt texts from dataset
            modified_chat_prompts = [self.tokenizer.apply_chat_template(
                text, 
                add_generation_prompt=True, 
                tokenize=False
            ) for text in batch.non_tensor_batch["modified_prompt_texts"]]
            
            # For debug info, also get original prompts
            # raw_prompt_ids_list = batch.non_tensor_batch["raw_prompt_ids"]
            # original_chat_prompts = [
            #     self.tokenizer.decode(ids, skip_special_tokens=False) 
            #     for ids in raw_prompt_ids_list
            # ]
        else:
            # Dynamic modification (original logic) - compute modified prompts
            raw_prompt_ids_list = batch.non_tensor_batch["raw_prompt_ids"]
            
            # Decode to get full chat template text (with all special tokens)
            original_chat_prompts = [
                self.tokenizer.decode(ids, skip_special_tokens=False) 
                for ids in raw_prompt_ids_list
            ]
            extra_info = [item for item in batch.non_tensor_batch["extra_info"]]
            modified_chat_prompts = []
            for i, chat_prompt in enumerate(original_chat_prompts):
                # Extract and modify messages (prompt_modifier will handle any modifications)
                messages = self._extract_chat_messages(chat_prompt, extra_info[i])
                
                # Rebuild using tokenizer.apply_chat_template
                modified_chat_prompt = self._rebuild_chat_prompt(messages)
                modified_chat_prompts.append(modified_chat_prompt)
        
        # Use verl_F.tokenize_and_postprocess_data for cleaner processing
        import verl.utils.torch_functional as verl_F
        
        modified_prompt_ids_list = []
        modified_attention_mask_list = []
        for prompt_with_chat_template in modified_chat_prompts:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.get("truncation", "error")
            )
            modified_prompt_ids_list.append(input_ids)
            modified_attention_mask_list.append(attention_mask)
        
        # Stack to create batch tensors
        modified_prompt_ids = torch.stack(modified_prompt_ids_list, dim=0).squeeze().to(prompt_ids.device)
        modified_attention_mask = torch.stack(modified_attention_mask_list, dim=0).squeeze().to(prompt_ids.device)

        # Simple direct concatenation since both are already padded tensors
        combined_input_ids = torch.cat([modified_prompt_ids, response_ids], dim=1)
        
        # Extract response attention mask from original batch using prompt length info
        original_attention_mask = batch.batch["attention_mask"]
        original_prompt_length = prompt_ids.shape[1]  # Original prompt length
        
        # Extract response attention mask (everything after original prompt)
        response_attention_mask = original_attention_mask[:, original_prompt_length:]
        combined_attention_mask = torch.cat([modified_attention_mask, response_attention_mask], dim=1)
        

        # Update batch with concatenated sequences
        modified_batch.batch["input_ids"] = combined_input_ids
        modified_batch.batch["attention_mask"] = combined_attention_mask  
        modified_batch.batch["response_mask"] = response_attention_mask
        modified_batch.batch["prompts"] = modified_prompt_ids
        modified_batch.batch["responses"] = response_ids  # Keep original
        
        # Fix position_ids: preserve response structure, only modify prompt part
        if "position_ids" in batch.batch:
            from verl.utils.model import compute_position_id_with_mask
            
            # Get original position_ids and extract prompt/response parts
            original_position_ids = batch.batch["position_ids"]
            original_prompt_length = prompt_ids.shape[1]
            # Compute position_ids for modified prompt
            modified_prompt_position_ids = compute_position_id_with_mask(modified_attention_mask)
            
            # Extract response position_ids from original batch
            response_position_ids = original_position_ids[:, original_prompt_length:]
            
            # Get the last position from modified prompt to adjust response positions
            if modified_prompt_position_ids.sum(dim=1).max() > 0:  # Check if there are valid tokens
                # Find the last valid position in each sequence
                last_prompt_positions = []
                for i in range(modified_prompt_position_ids.shape[0]):
                    valid_mask = modified_attention_mask[i] > 0
                    if valid_mask.any():
                        last_pos = modified_prompt_position_ids[i][valid_mask][-1]
                        last_prompt_positions.append(last_pos.item())
                    else:
                        last_prompt_positions.append(-1)
                
                # Adjust response position_ids to continue from modified prompt
                adjusted_response_position_ids = response_position_ids.clone()
                for i in range(response_position_ids.shape[0]):
                    if last_prompt_positions[i] >= 0:
                        # Shift response positions to continue from last prompt position
                        response_mask = response_attention_mask[i] > 0
                        if response_mask.any():
                            # Find first valid response position to calculate offset
                            first_response_pos = response_position_ids[i][response_mask][0].item()
                            offset = last_prompt_positions[i] + 1 - first_response_pos
                            adjusted_response_position_ids[i] += offset
                            
                response_position_ids = adjusted_response_position_ids
            
            # Concatenate modified prompt position_ids with adjusted response position_ids
            position_ids = torch.cat([modified_prompt_position_ids, response_position_ids], dim=1)
            modified_batch.batch["position_ids"] = position_ids
        
        # Store debug information as numpy arrays (required for DataProto chunking)
        # modified_batch.non_tensor_batch["original_prompt_texts"] = np.array(original_chat_prompts, dtype=object)
        modified_batch.non_tensor_batch["modified_prompt_texts"] = np.array(modified_chat_prompts, dtype=object)
        modified_batch.meta_info["prompt_modified"] = True
        # modified_batch.meta_info["original_prompt_lengths"] = [len(p) for p in original_chat_prompts]
        modified_batch.meta_info["modified_prompt_lengths"] = [len(p) for p in modified_chat_prompts]
        
        # Log one example of modified prompt for debugging
        if len(modified_chat_prompts) > 0:
            print(f"[DEBUG] Modified prompt example:\n{modified_chat_prompts[0]}")
            print(f"[DEBUG] Total modified prompts: {len(modified_chat_prompts)}")
            if "modified_prompt_texts" in batch.non_tensor_batch:
                print("[DEBUG] Using pre-computed modified_prompt_texts from dataset")
            else:
                print("[DEBUG] Dynamically generated modified prompts")
        return modified_batch
    
    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """
        Override reference log prob computation to use modified prompts.
        """
        # Apply prompt modification before computing reference log probs
        modified_batch = self._modify_prompts_for_ref_policy(batch)
        
        # Use the parent class method with modified batch
        return super()._compute_ref_log_prob(modified_batch)
    
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # evaluate using reward_function
            if not self.use_reward_loop:
                reward_tensor, reward_extra_info = self._compute_reward_legacy(
                    test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
                )
            else:
                reward_tensor = test_batch.batch["rm_scores"]
                reward_extra_keys = test_batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: test_batch.non_tensor_batch[key] for key in reward_extra_keys}
            
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if reward_extra_info:
                for key, lst in reward_extra_info.items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}

        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            # Define core classification metrics that should be in val-core
            core_classification_vars = {"precision", "recall", "f1_score", "accuracy", "balanced_accuracy", "specificity"}
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var or var_name in core_classification_vars)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict
    
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            if not self.use_reward_loop:
                                reward_baseline_tensor = self._compute_reward_legacy(
                                    batch, reward_fn=self.reward_fn, sum_reward=True
                                )
                            else:
                                reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # Compute or extract reward_tensor and reward_extra_infos_dict for training
                        if not self.use_reward_loop:
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = self._compute_reward_legacy(
                                    batch, reward_fn=self.reward_fn, reward_for_val=False
                                )
                        else:
                            reward_tensor = batch.batch["rm_scores"]
                            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                            reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            
                            reward_metrics = {}
                            for key, values in reward_extra_infos_dict.items():
                                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                                    arr = np.array(values)
                                    reward_metrics[f"reward/{key}/mean"] = np.mean(arr).item()
                                    reward_metrics[f"reward/{key}/max"] = np.max(arr).item()
                                    reward_metrics[f"reward/{key}/min"] = np.min(arr).item()
                            metrics.update(reward_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)