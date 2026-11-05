# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import os
import time
import json
import regex as re
import numpy as np
from pathlib import Path
from verl import DataProto
from .reward_score import _default_compute_score
from .reward_score.torl_math import compute_score as torl_compute_score
from .reward_score.math import compute_score as math_compute_score
from .reward_score.msearthmcq import compute_score_msearthmcq
from .reward_score.chartqa import compute_score_chartqa
from .reward_score.vstar import compute_score_vstar
from .reward_score.slake import compute_score_slake
from .reward_score.zwz import compute_score_zwz
from verl.workers.reward_manager import register
from router_r1.llm_agent.route_service import check_llm_name, check_tool_name
import torch
from collections import defaultdict

def extract_box_contents(text):
    """
    Extract contents from \box{} commands in a string.
    
    Args:
        text (str): Input string containing \box{} commands
        
    Returns:
        list: List of contents found inside \box{} commands
    """
    # Pattern to match \box{content} with proper brace matching
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    # Find all matches
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""

def extract_answer(text, mode='math'):
    """
    Extract the final answer from the text based on the mode.
    
    Args:
        text (str): Input string containing the answer
        mode (str): Mode of extraction ('math' or 'lcb_code')
    Returns:
        str: Extracted answer
    """
    if mode == 'math':
        return extract_box_contents(text)
    elif mode == 'lcb_code':
        start_idx = text.rfind('```python')
        if start_idx != -1:
            end_idx = text.find('```', start_idx+len('```python'))
            if end_idx != -1:
                end_idx += len('```')
                return text[start_idx:end_idx].strip()
            else:
                return text[start_idx:].strip()
        else:
            if text.startswith("#"):
                # this is alread the pure code, but without ```python
                return "```python\n" + text.strip() + "\n```"
            else:
                return ""
    elif mode == 'hle_judge':
        return text
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
@register("torl")
class ToRLRewardManager:
    """The reward manager.
    """
    name="torl"

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.compute_score = compute_score if compute_score else _default_compute_score
        self.compute_score = torl_compute_score
        self.reward_fn_key = reward_fn_key

        # Mapping of data_source to reward functions
        self.reward_function_map = {
            # Use math functions for geometry and counting tasks
            'geometry3k': math_compute_score,
            'tallyqa': math_compute_score,
            # Use msearthmcq functions for earth science and vision tasks
            'msearthmcq': compute_score_msearthmcq,
            'microvqa': compute_score_msearthmcq,
            # Use chartqa functions for chart-related tasks
            'chartqa': compute_score_chartqa,
            # Use slake functions for medical tasks
            'slake': compute_score_slake,
            'vstar': compute_score_vstar,
            'zwz': compute_score_zwz
        }
        self.step = None
        self.add_format_think_penalty = False # -0.5 if not begines with <think> and end with </think>
        self.add_format_answer_penalty = False # -0.5 if not having <answer> </answer>
        self.add_format_reward = True # Enable Router-R1 style format reward
        self.add_valid_action_penalty = False # -0.25 if num turns > 0 not action not valid
        self.add_unfinished_traj_penalty = False # -0.25 if the traj is not finished
        self.add_no_tool_interact_penalty = False # -0.25 if the traj's num turn is 0, no interaction at all
        self.add_code_exec_penalty = False # -0.25 if the execution has an error.

    def compute_format_reward(self, completion: str) -> float:
        """
        Compute format reward based on Router-R1 format_reward function.
        Returns: 0.0 (correct format) or negative penalty values.
        """
        PUNISH_REWARD_MAX = -1
        PUNISH_REWARD_MEDIUM = -1
        PUNISH_REWARD_SMALL = -1

        tag_enclose_pattern = r'<(search|answer|information)>(.*?)</\1>'
        tag_enclose_matches = re.findall(tag_enclose_pattern, completion, re.DOTALL)
        if len(tag_enclose_matches) == 0:
            # print(completion)
            # print("punish 1")
            return PUNISH_REWARD_MAX

        # if completion.count("<search>") != completion.count("</search>") or \
        if completion.count("<answer>") != completion.count("</answer>") or \
           completion.count("<information>") != completion.count("</information>"):
            # print(completion)
            # print("punish 2")
            return PUNISH_REWARD_MAX

        route_enclose_count = 0
        answer_enclose_count = 0
        think_enclose_count = 0
        info_enclose_count = 0
        is_nesting = False
        query_format_punish = False
        llm_name_punish = False
        tool_name_punish = False
        think_punish = False

        for single_match in tag_enclose_matches:
            action = single_match[0].strip()
            content = single_match[1].strip()
            if action == "search":
                route_enclose_count += 1
                # Check for new format: Model-Name@@Tool-Name:Your-Input
                if "@@" in content and ":" in content:
                    parts = content.split(":", 1)
                    if len(parts) == 2:
                        model_tool_part = parts[0].strip()
                        tool_input = parts[1].strip()

                        if "@@" in model_tool_part:
                            model_name, tool_name = model_tool_part.split("@@", 1)
                            model_name = model_name.strip()
                            tool_name = tool_name.strip()
                            llm_name_valid, _, _ = check_llm_name(model_name)
                            tool_name_valid = check_tool_name(tool_name)
                            if llm_name_valid == "":
                                llm_name_punish = True
                            # Validate format - simplified for verl-tool
                            if not model_name or not tool_name or not tool_input:
                                query_format_punish = True
                            elif "model-name" in model_name.lower() or "tool-name" in tool_name.lower() or "your-input" in tool_input.lower():
                                query_format_punish = True
                            # Skip LLM and tool name validation for verl-tool compatibility
                        else:
                            query_format_punish = True
                    else:
                        query_format_punish = True
                # Check for old format: LLM-Name:Query (for backward compatibility)
                elif ":" in content:
                    if content.count(":") == 1:
                        if content.split(":")[-1].strip() == '' or "llm-name" in content.strip().lower() \
                                or "your-query" in content.strip().lower() or content.split(":")[0].strip().lower() in content.split(":")[-1].strip().lower():
                            query_format_punish = True
                        # Skip LLM name validation for verl-tool compatibility
                    else:
                        query_format_punish = True
                else:
                    query_format_punish = True
            elif action == "answer":
                answer_enclose_count += 1
            elif action == "think":
                think_enclose_count += 1
                if content == "..." or content == "":
                    think_punish = True
            else:
                info_enclose_count += 1

            if content.count("<search>") + content.count("</search>") + content.count("<think>") + content.count("</think>") + content.count("<answer>") + content.count("</answer>") + content.count("<information>") + content.count("</information>") != 0:
                is_nesting = True

        # if think_punish:
        #     return PUNISH_REWARD_MAX

        # if is_nesting:
        #     return PUNISH_REWARD_MAX

        # Strict requirement: responses MUST end with </answer>
        # if not completion.endswith("</answer>"):
        #     return PUNISH_REWARD_MAX

        # If ends with </answer>, must have exactly one answer tag and at least one think tag
        # Allow flexible search/info count (can be 0 if direct answer)
        if answer_enclose_count != 1 or route_enclose_count < 1:
            # print(completion)
            # print("punish 3")
            return PUNISH_REWARD_MAX

        completion = completion.strip()
        # if not completion[:len("<think>")] == "<think>":
        #     print(completion)
        #     print("punish 4")
        #     return PUNISH_REWARD_MAX

        # Strict requirement: responses MUST end with </answer>
        # if not completion.endswith("</answer>"):
        #     return PUNISH_REWARD_MAX

        if query_format_punish:
            # print(completion)
            # print("punish 5")
            return PUNISH_REWARD_MEDIUM

        if llm_name_punish or tool_name_punish:
            # print(completion)
            # print("punish 6")
            return PUNISH_REWARD_SMALL


        return 0.0

    def add_additional_penalties(self, response: str, data_i, scores_i: dict):
        # Router-R1 style format reward
        if self.add_format_reward:
            format_penalty = self.compute_format_reward(response)
            # format_penalty = 0.0
            import random   
            if random.randint(1, 32) == 1:
                print(f"format reward: {format_penalty}")
            scores_i['score'] += format_penalty  # format_penalty is already negative for penalties
            scores_i['format_reward'] = format_penalty

        # 1.4 format penalty
        # if self.add_format_think_penalty:
        #     match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        #     if not match or not response.startswith("<think>") or response.count("<think>") != 1 or response.count("</think>") != 1:
        #         scores_i['score'] -= 0.5
        #         scores_i['think_format_penalty'] = 1
        #     else:
        #         scores_i['think_format_penalty'] = 0
        # if self.add_format_answer_penalty:
        #     match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        #     if not match or not response.endswith("</answer>") or response.count("<answer>") != 1 or response.count("</answer>") != 1:
        #         scores_i['score'] -= 0.5
        #         scores_i['answer_format_penalty'] = 1
        #     else:
        #         scores_i['answer_format_penalty'] = 0
        if "turns_stats" in data_i.non_tensor_batch:
            if self.add_valid_action_penalty:
                num_turn = data_i.non_tensor_batch["turns_stats"]
                num_valid_action = data_i.non_tensor_batch["valid_action_stats"]
                if num_valid_action < num_turn:
                    scores_i['score'] -= 0.25
                    scores_i['valid_action_penalty'] = 1
                else:
                    scores_i['valid_action_penalty'] = 0
            if self.add_unfinished_traj_penalty:
                is_active = data_i.non_tensor_batch["active_mask"]
                if is_active:
                    scores_i['score'] -= 0.25
                    scores_i['unfinished_traj_penalty'] = 1
                else:
                    scores_i['unfinished_traj_penalty'] = 0
            if self.add_no_tool_interact_penalty:
                num_valid_action = data_i.non_tensor_batch["valid_action_stats"]
                if num_valid_action == 0:
                    scores_i['score'] -= 0.25
                    scores_i['no_tool_interact_penalty'] = 1
                else:
                    scores_i['no_tool_interact_penalty'] = 0
            if self.add_code_exec_penalty:
                keywords = ["ERROR:\nTraceback", "Execution timed out"]
                if any(keyword in response for keyword in keywords):
                    scores_i['score'] -= 0.25
                    scores_i['exec_error'] = 1
                else:
                    scores_i['exec_error'] = 0
        
        return scores_i
    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # check the last step index
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            score = {}
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            extracted_answer = extract_answer(response_str, mode='math')

            # Select reward function based on data_source
            reward_function = self.reward_function_map.get(data_source, self.compute_score)

            reward_result = reward_function(
                solution_str=response_str,
                ground_truth=ground_truth,
            )

            # Handle different return formats
            if data_source in self.reward_function_map:
                # New reward functions return tuples
                if isinstance(reward_result, tuple):
                    if len(reward_result) >= 3:
                        # For most functions: (metric_score, cost_score, reward_score)
                        # We use the last element (reward_score) for comparison
                        torl_score = reward_result[-1]  # Use the final reward score
                    else:
                        torl_score = reward_result[0]  # Fallback to first element
                else:
                    torl_score = reward_result
            else:
                # Original torl_compute_score returns single value
                torl_score = reward_result
            # # 将结果奖励设置为0
            # torl_score = 0
            # Ensure torl_score is a number for comparison
            if isinstance(torl_score, (int, float)):
                score['accuracy'] = 1. if torl_score > 0 else 0.
                score['score'] = torl_score
            else:
                # If still not a number, try to extract a numeric value
                try:
                    numeric_score = float(torl_score)
                    score['accuracy'] = 1. if numeric_score > 0 else 0.
                    score['score'] = numeric_score
                except (ValueError, TypeError):
                    # Fallback: treat as incorrect
                    score['accuracy'] = 0.
                    score['score'] = -1.0
            score['has_answer'] = 1. if extracted_answer else 0.

            # add additional penalty
            score = self.add_additional_penalties(response_str, data_item, score)      

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = score["accuracy"] # for validation
            else:
                if self.num_examine == 1:
                    reward = score if score > 0 else 0.0
                else:
                    reward = score

            reward_tensor[i, valid_response_length - 1] = reward 

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
                
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else None
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else None
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items()))
            }
        else:
            return reward_tensor
