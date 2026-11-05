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

import re
import string


def extract_answer(text):
    """Extract answer from the solution text using various regex patterns."""
    # Common patterns for answer extraction
    patterns = [
        r'<answer>(.*?)</answer>',  # 新增：匹配 <answer> 标签
        r'Answer:\s*(.*)',
        r'answer:\s*(.*)',
        r'ANSWER:\s*(.*)',
        r'Final Answer:\s*(.*)',
        r'final answer:\s*(.*)',
        r'FINAL ANSWER:\s*(.*)',
        r'\\boxed\{([^}]*)\}',
        r'\$\$(.*?)\$\$',
        r'\$(.*?)\$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = answer.strip(string.punctuation + string.whitespace)

            # 如果是从 <answer> 标签中提取的，对内容进行进一步处理
            if pattern == r'<answer>(.*?)</answer>':
                # 对 <answer> 标签内的内容，使用其他模式进行二次提取
                inner_patterns = [
                    r'The answer is\s+(.*)',  # 新增：匹配 "The answer is ..."
                    r'Answer:\s*(.*)',
                    r'answer:\s*(.*)',
                    r'ANSWER:\s*(.*)',
                    r'Final Answer:\s*(.*)',
                    r'final answer:\s*(.*)',
                    r'FINAL ANSWER:\s*(.*)',
                    r'\\boxed\{([^}]*)\}',
                    r'\$\$(.*?)\$\$',
                    r'\$(.*?)\$',
                ]

                for inner_pattern in inner_patterns:
                    inner_match = re.search(inner_pattern, answer, re.IGNORECASE)
                    if inner_match:
                        inner_answer = inner_match.group(1).strip()
                        inner_answer = inner_answer.strip(string.punctuation + string.whitespace)
                        return inner_answer

                # 如果没有找到内部模式，直接返回清理后的内容
                return answer

            return answer

    # If no pattern matches, try to find the last line that looks like an answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(('Question:', 'Problem:', 'Solve:', 'Given:')):
            # Remove common prefixes that might appear in the answer line
            line = re.sub(r'^(So|Therefore|Thus|Hence|Finally|The answer is)\s*[:\-]?\s*', '', line, flags=re.IGNORECASE)
            line = line.strip(string.punctuation + string.whitespace)
            if line:
                return line

    return ""


def compute_score(solution_str, ground_truth, format_score=0.0, score=1., cost_coe=0.0, api_cost=0.0, state="train", reward_metric="f1", return_answer=False):
    """
    Compute score for MMLU-style multiple choice questions.

    Args:
        solution_str: The model's response
        ground_truth: Dictionary containing the ground truth information
        format_score: Score penalty for format issues (not used in this implementation)
        score: Score for correct answer (default 1.0)
        cost_coe: Cost coefficient
        api_cost: API cost
        state: "train" or "val"
        reward_metric: Not used for msearthmcq (always binary 0/1)
        return_answer: If True, return tuple of (score_info, extracted_answer)

    Returns:
        If return_answer is False:
            For train: (metric_score, cost_score, reward_score)
            For val: (metric_em, metric_f1, cost_score, reward_score)
        If return_answer is True:
            (score_info, extracted_answer) where score_info is the original return value
    """
    try:
        # Extract answer from solution
        have_answer_tag = ('<answer>' in solution_str and '</answer>' in solution_str)
        extracted_answer = extract_answer(solution_str)

        if not extracted_answer:
            accuracy_score = 0.0
            answer = None
        else:
            # Get target answers from ground truth
            target = ground_truth
            if len(target) < 2:
                accuracy_score = 0.0
                answer = extracted_answer
            else:
                extracted_lower = extracted_answer.lower()
                target_0_lower = target[0].lower() if target[0] else ""
                target_1_lower = target[1].lower() if target[1] else ""

                # Check if the answer is correct
                is_correct = False

                if len(extracted_answer) == 1:
                    # Single character answer - exact match with first target
                    is_correct = extracted_lower == target_0_lower
                else:
                    # Multi-character answer - substring match with second target
                    is_correct = (extracted_lower in target_1_lower) or (target_1_lower in extracted_lower)

                accuracy_score = 1.0 if is_correct else 0.0
                answer = extracted_answer

        if not have_answer_tag:
            accuracy_score = 0.0

        # Debug print (similar to math.py)
        import random   
        if random.randint(1, 64) == 1:  # 可以改为随机或条件打印
            print(f"[debug msearthmcq] solution_str: {solution_str}")
            print(f"[debug msearthmcq] target: {target if 'target' in locals() else 'N/A'}")
            print(f"[debug msearthmcq] answer: {answer}")
            if answer is not None and 'target' in locals() and len(target) >= 2:
                print(f"[debug msearthmcq] is_correct: {accuracy_score}")
            print(f"[debug msearthmcq] accuracy_score: {accuracy_score}")

        if return_answer:
            # Return both score and extracted answer
            if state == "train":
                if format_score == -1.0:
                    score_info = (accuracy_score, api_cost, format_score)
                else:
                    if accuracy_score == 0:
                        score_info = (accuracy_score, api_cost, accuracy_score + format_score)
                    else:
                        score_info = (accuracy_score, api_cost, (accuracy_score + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)
            else:
                metric_em = accuracy_score
                metric_f1 = accuracy_score
                if format_score == -1.0:
                    score_info = (metric_em, metric_f1, api_cost, format_score)
                else:
                    if accuracy_score == 0:
                        score_info = (metric_em, metric_f1, api_cost, accuracy_score + format_score)
                    else:
                        score_info = (metric_em, metric_f1, api_cost, (accuracy_score + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)
            return score_info, answer
        else:
            # Original return behavior
            if state == "train":
                if format_score == -1.0:
                    return accuracy_score, api_cost, format_score
                else:
                    if accuracy_score == 0:
                        return accuracy_score, api_cost, accuracy_score + format_score
                    else:
                        return accuracy_score, api_cost, (accuracy_score + format_score) * (1.0 - cost_coe) + api_cost * cost_coe
            else:
                metric_em = accuracy_score
                metric_f1 = accuracy_score
                if format_score == -1.0:
                    return metric_em, metric_f1, api_cost, format_score
                else:
                    if accuracy_score == 0:
                        return metric_em, metric_f1, api_cost, accuracy_score + format_score
                    else:
                        return metric_em, metric_f1, api_cost, (accuracy_score + format_score) * (1.0 - cost_coe) + api_cost * cost_coe

    except Exception as e:
        print(f"Error in compute_score for msearthmcq: {e}")
        # Return 0 scores on error
        if return_answer:
            if state == "train":
                score_info = (0.0, api_cost, format_score if format_score == -1.0 else 0.0)
            else:
                score_info = (0.0, 0.0, api_cost, format_score if format_score == -1.0 else 0.0)
            return score_info, solution_str
        else:
            if state == "train":
                return 0.0, api_cost, format_score if format_score == -1.0 else 0.0
            else:
                return 0.0, 0.0, api_cost, format_score if format_score == -1.0 else 0.0


def compute_score_msearthmcq(solution_str, ground_truth, format_score=0.0, **kwargs):
    """
    Alias for compute_score function for backward compatibility.
    """
    return compute_score(solution_str, ground_truth, format_score=format_score, **kwargs)


if __name__ == "__main__":
    import numpy as np
    print("=" * 60)
    print("Testing msearthmcq.py functions")
    print("=" * 60)

    # Test case 1: Single character answer
    print("\nTest case 1 - Single character answer:")
    print("-" * 45)
    solution_str = "<answer>The answer is A</answer>"
    ground_truth = ["A", "Single character"]
    train_metric, train_cost, train_reward = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        state="train",
        format_score=1.0,  # 假设格式完全正确
        api_cost=0.0,
        cost_coe=0.0,
    )
    print(f"Solution string: {solution_str}")
    print(f"Ground truth:   {ground_truth}")
    print(f"Train metric:   {train_metric}")
    print(f"Train cost:     {train_cost}")
    print(f"Train reward:   {train_reward}")

    # Test case 2: Multi-character answer with substring matching
    print("\nTest case 2 - Multi-character answer (substring matching):")
    print("-" * 60)
    solution_str = "<answer>The answer is Northeast to Southwest</answer>"
    ground_truth = ["B", "Northeast to Southwest direction"]
    train_metric, train_cost, train_reward = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        state="train",
        format_score=1.0,
        api_cost=0.0,
        cost_coe=0.0,
    )
    print(f"Solution string: {solution_str}")
    print(f"Ground truth:   {ground_truth}")
    print(f"Train metric:   {train_metric}")
    print(f"Train cost:     {train_cost}")
    print(f"Train reward:   {train_reward}")

    # Test case 3: Wrong answer
    print("\nTest case 3 - Wrong answer:")
    print("-" * 30)
    solution_str = "<answer>The answer is C</answer>"
    ground_truth = ["A", "Single character"]
    train_metric, train_cost, train_reward = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        state="train",
        format_score=0.5,  # 格式部分正确
        api_cost=0.0,
        cost_coe=0.0,
    )
    print(f"Solution string: {solution_str}")
    print(f"Ground truth:   {ground_truth}")
    print(f"Train metric:   {train_metric}")
    print(f"Train cost:     {train_cost}")
    print(f"Train reward:   {train_reward}")

    # Test case 4: No answer extracted
    print("\nTest case 4 - No answer extracted:")
    print("-" * 35)
    solution_str = "This is just some text without a clear answer."
    ground_truth = ["A", "Single character"]
    train_metric, train_cost, train_reward = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        state="train",
        format_score=0.0,
        api_cost=0.0,
        cost_coe=0.0,
    )
    print(f"Solution string: {solution_str}")
    print(f"Ground truth:   {ground_truth}")
    print(f"Train metric:   {train_metric}")
    print(f"Train cost:     {train_cost}")
    print(f"Train reward:   {train_reward}")

    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)
