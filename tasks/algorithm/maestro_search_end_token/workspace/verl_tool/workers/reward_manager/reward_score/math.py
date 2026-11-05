# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
import random
import numpy as np
import sympy as sp


def latex_to_float(expr):
    """
    将 latex / 数学表达式转换为 float
    支持 sqrt / frac
    """
    if expr is None:
        return None

    try:
        expr = strip_string(expr)

        # frac -> (a)/(b)
        expr = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', expr)

        # sqrt -> sqrt()
        expr = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', expr)

        # 移除多余符号
        expr = expr.replace("^", "**")

        val = float(sp.N(sp.sympify(expr)))
        return val
    except Exception:
        return None

def extract_answer_tag(solution_str: str) -> str:
    """
    Extract content from <answer></answer> tag if present.
    Returns the extracted content, or None if not found.
    """
    if not solution_str:
        return None
    
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    matches = list(match)
    
    # If there are matches, return the last one (most recent answer)
    if len(matches) > 0:
        return matches[-1].group(1).strip()
    
    return None


def extract_answer_from_text(text):
    """
    从文本中提取答案，尝试匹配常见的答案格式
    
    Args:
        text: 输入文本
        
    Returns:
        提取到的答案字符串，如果没有匹配到则返回None
    """
    if not text:
        return None
    
    # 常见的答案格式模式（不区分大小写）
    patterns = [
        r'the\s+answer\s+is\s*:?\s*(.+?)(?:\.|$|\\n|\\r)',
        r'answer\s*:?\s*(.+?)(?:\.|$|\\n|\\r)',
        r'the\s+final\s+answer\s+is\s*:?\s*(.+?)(?:\.|$|\\n|\\r)',
        r'final\s+answer\s*:?\s*(.+?)(?:\.|$|\\n|\\r)',
        r'solution\s*:?\s*(.+?)(?:\.|$|\\n|\\r)',
        r'the\s+solution\s+is\s*:?\s*(.+?)(?:\.|$|\\n|\\r)',
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # 移除可能的引号
            answer = answer.strip('"\'')
            # 移除可能的句号结尾
            answer = answer.rstrip('.')
            if answer:
                return answer
    
    return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.,
                 cost_coe=0.0, api_cost=0.0, state="train", reward_metric="f1", return_answer=False):
    """
    Compute score for math problems (using \\boxed{...} format).

    Args:
        solution_str: The full model output string
        ground_truth: Dict containing 'target' key, or direct string/list of answers
        format_score: Score for format correctness
        score: Score for correct answer (default 1.0)
        cost_coe: Cost coefficient
        api_cost: API cost
        state: "train" or "val"
        reward_metric: Not used for math (always binary 0/1)
        return_answer: If True, return tuple of (score_info, extracted_answer)

    Returns:
        If return_answer is False:
            For train: (metric_score, cost_score, reward_score)
            For val: (metric_em, metric_f1, cost_score, reward_score)
        If return_answer is True:
            (score_info, extracted_answer) where score_info is the original return value
    """
    # Priority 0: Check if there's <answer></answer> tag, extract its content
    have_answer_tag = ('<answer>' in solution_str and '</answer>' in solution_str)
    answer_tag_content = extract_answer_tag(solution_str)
    if answer_tag_content is not None:
        solution_str = answer_tag_content
    
    # Extract ground truth target
    if isinstance(ground_truth, dict):
        target = ground_truth.get('target', '')
    else:
        target = ground_truth
    
    # Handle numpy array
    if isinstance(target, np.ndarray):
        target = target.tolist()
    elif hasattr(target, '__iter__') and not isinstance(target, (str, bytes)):
        # Handle other array-like types (e.g., pandas Series)
        try:
            target = list(target)
        except (TypeError, ValueError):
            target = [target] if target is not None else ''
    
    if isinstance(target, list):
        target = target[0] if target else ''
    
    retval = 0.
    answer = None
    try:
        # import pdb; pdb.set_trace()
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            # 如果找到了boxed，提取其中的内容
            answer = remove_boxed(string_in_last_boxed)
        else:
            # 如果没有找到boxed，尝试匹配常见的答案格式
            extracted_answer = extract_answer_from_text(solution_str)
            if extracted_answer is not None:
                answer = extracted_answer
            else:
                # 如果都没有匹配到，直接使用整个solution_str作为answer
                answer = solution_str.strip()
        
        # 比较answer和target
        if answer and is_equiv(answer, target):
            retval = 1.
    except Exception as e:
        if random.randint(1, 64) == 1:
            print(e)
        # 如果处理过程中出错，尝试直接使用solution_str
        if answer is None:
            answer = solution_str.strip()

    if not have_answer_tag:
        metric = 0.0

    metric = retval
    # if random.randint(1, 20) == 1:
    if random.randint(1, 64) == 1:
    # if True:
        print(f"[debug] solution_str: {solution_str}")
        print(f"[debug] target: {target}")
        print(f"[debug] answer: {answer}")
        if answer is not None:
            print(f"[debug] is_equiv: {is_equiv(answer, target)}")
        print(f"[debug] metric: {metric}")
    if return_answer:
        # Return both score and extracted answer
        if state == "train":
            if format_score == -1.0:
                score_info = (metric, api_cost, format_score)
            else:
                if metric == 0:
                    score_info = (metric, api_cost, metric + format_score)
                else:
                    score_info = (metric, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)
        else:
            metric_em = metric
            metric_f1 = metric
            if format_score == -1.0:
                score_info = (metric_em, metric_f1, api_cost, format_score)
            else:
                if metric == 0:
                    score_info = (metric_em, metric_f1, api_cost, metric + format_score)
                else:
                    score_info = (metric_em, metric_f1, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)
        return score_info, answer
    else:
        # Original return behavior
        if state == "train":
            if format_score == -1.0:
                return metric, api_cost, format_score
            else:
                if metric == 0:
                    return metric, api_cost, metric + format_score
                else:
                    return metric, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe
        else:
            metric_em = metric
            metric_f1 = metric
            if format_score == -1.0:
                return metric_em, metric_f1, api_cost, format_score
            else:
                if metric == 0:
                    return metric_em, metric_f1, api_cost, metric + format_score
                else:
                    return metric_em, metric_f1, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    # import pdb; pdb.set_trace()
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)

        if ss1 == ss2:
            return True

        # 数值计算
        n1 = latex_to_float(ss1)
        n2 = latex_to_float(ss2)
        # import pdb; pdb.set_trace()
        if n1 is not None and n2 is not None:
            return round(n1, 2) == round(n2, 2)

        return ss1 == ss2

    except Exception:
        return False


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    
    new_string = splits[0]
    for split in splits[1:]:
        if split.startswith("{"):
            new_substr = "\\sqrt" + split
        elif len(split) > 0:
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            return string
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # -------- 新增：规范 sqrt / frac 空格 --------

    # \sqrt { 3 } -> \sqrt{3}
    string = re.sub(r'\\sqrt\s*\{\s*([^}]*)\s*\}', r'\\sqrt{\1}', string)

    # \frac { a } { b } -> \frac{a}{b}
    string = re.sub(
        r'\\frac\s*\{\s*([^}]*)\s*\}\s*\{\s*([^}]*)\s*\}',
        r'\\frac{\1}{\2}',
        string
    )

    # --------------------------------------------

    # " 0." -> "0."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string

    if string[0] == ".":
        string = "0" + string

    # remove variable assignment like k = ...
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # remove spaces
    string = string.replace(" ", "")

    # fix sqrt3 -> sqrt{3}
    string = fix_sqrt(string)

    # fix fractions
    string = fix_fracs(string)

    # manually change 0.5
    if string == "0.5":
        string = "\\frac{1}{2}"

    # convert a/b -> \frac{a}{b}
    string = fix_a_slash_b(string)

    return string


if __name__ == "__main__":
    print("=" * 60)
    print("Testing math.py functions")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Simple example: compare one model output string with a ground truth
    # ------------------------------------------------------------------
    example_solution = "<answer><|begin_of_box|>\\boxed{4}<|end_of_box|></answer>"
    example_ground_truth = {"target": "4"}
    train_metric, train_cost, train_reward = compute_score(
        solution_str=example_solution,
        ground_truth=example_ground_truth,
        state="train",
        format_score=0.0,  # 假设格式完全正确
        api_cost=0.0,
        cost_coe=0.0,
    )
    print("\nSimple compute_score example:")
    print("-" * 60)
    print(f"Solution string: {example_solution}")
    print(f"Ground truth:   {example_ground_truth}")
    print(f"Train metric:   {train_metric}")
    print(f"Train cost:     {train_cost}")
    print(f"Train reward:   {train_reward}")

    # # Test last_boxed_only_string
    # print("\n1. Testing last_boxed_only_string:")
    # print("-" * 60)
    # boxed_tests = [
    #     ("The answer is \\boxed{42}", "\\boxed{42}"),
    #     ("\\boxed{x^2 + y^2}", "\\boxed{x^2 + y^2}"),
    #     ("First \\boxed{1} then \\boxed{2}", "\\boxed{2}"),  # Should get last one
    #     ("\\boxed{3.14}", "\\boxed{3.14}"),
    #     ("\\boxed {42}", "\\boxed {42}"),  # With space
    #     ("No boxed here", None),
    #     ("\\boxed{nested{inner}}", "\\boxed{nested{inner}}"),
    # ]
    
    # for i, (input_text, expected) in enumerate(boxed_tests, 1):
    #     result = last_boxed_only_string(input_text)
    #     status = "✓" if result == expected else "✗"
    #     print(f"  Test {i}: {status}")
    #     print(f"    Input:    {repr(input_text)}")
    #     print(f"    Expected: {repr(expected)}")
    #     print(f"    Got:      {repr(result)}")
    #     if result != expected:
    #         print(f"    ⚠️  MISMATCH!")
    
    # # Test remove_boxed
    # print("\n2. Testing remove_boxed:")
    # print("-" * 60)
    # remove_tests = [
    #     ("\\boxed{42}", "42"),
    #     ("\\boxed {3.14}", "3.14"),
    #     ("\\boxed{x^2}", "x^2"),
    # ]
    
    # for i, (input_text, expected) in enumerate(remove_tests, 1):
    #     try:
    #         result = remove_boxed(input_text)
    #         status = "✓" if result == expected else "✗"
    #         print(f"  Test {i}: {status}")
    #         print(f"    Input:    {repr(input_text)}")
    #         print(f"    Expected: {repr(expected)}")
    #         print(f"    Got:      {repr(result)}")
    #         if result != expected:
    #             print(f"    ⚠️  MISMATCH!")
    #     except Exception as e:
    #         print(f"  Test {i}: ✗ Error: {e}")
    #         print(f"    Input:    {repr(input_text)}")
    
    # # Test is_equiv
    # print("\n3. Testing is_equiv:")
    # print("-" * 60)
    # equiv_tests = [
    #     ("42", "42", True),
    #     ("3.14", "3.14", True),
    #     ("\\frac{1}{2}", "0.5", True),  # Should normalize
    #     ("x^2", "x^2", True),
    #     ("42", "43", False),
    #     ("1/2", "\\frac{1}{2}", True),  # Should normalize
    # ]
    
    # for i, (str1, str2, expected) in enumerate(equiv_tests, 1):
    #     result = is_equiv(str1, str2)
    #     status = "✓" if result == expected else "✗"
    #     print(f"  Test {i}: {status}")
    #     print(f"    str1:     {repr(str1)}")
    #     print(f"    str2:     {repr(str2)}")
    #     print(f"    Expected: {expected}")
    #     print(f"    Got:      {result}")
    #     if result != expected:
    #         print(f"    ⚠️  MISMATCH!")
    
    # # Test compute_score - full solution string
    # print("\n4. Testing compute_score (full solution extraction):")
    # print("-" * 60)
    # score_tests = [
    #     # (solution_str, ground_truth, expected_metric)
    #     ("The solution is \\boxed{42}", "42.0", 1.0),
    #     ("First \\boxed{1} then \\boxed{2}", "2", 1.0),  # Should use last boxed
    #     ("The answer is \\boxed{3.14}", "3.14", 1.0),
    #     ("The answer is \\boxed{42}", "43", 0.0),  # Wrong answer
    #     ("No boxed here", "42", 0.0),  # No boxed found
    #     ("\\boxed{\\frac{1}{2}}", "0.5", 1.0),  # Should normalize
    # ]
    
    # for i, (solution_str, ground_truth, expected_metric) in enumerate(score_tests, 1):
    #     # Test with dict format
    #     gt_dict = {"target": ground_truth}
    #     result_train = compute_score(solution_str, gt_dict, state="train")
    #     result_val = compute_score(solution_str, gt_dict, state="val")
        
    #     metric_train = result_train[0]
    #     metric_val = result_val[0]
        
    #     status = "✓" if metric_train == expected_metric and metric_val == expected_metric else "✗"
    #     print(f"  Test {i}: {status}")
    #     print(f"    Solution:  {repr(solution_str[:50])}...")
    #     print(f"    Ground:    {repr(ground_truth)}")
    #     print(f"    Expected:  {expected_metric}")
    #     print(f"    Train:     {result_train} (metric={metric_train})")
    #     print(f"    Val:       {result_val} (metric={metric_val})")
    #     if metric_train != expected_metric or metric_val != expected_metric:
    #         print(f"    ⚠️  MISMATCH!")
    
    # # Interactive test
    # print("\n" + "=" * 60)
    # print("Interactive test for compute_score")
    # print("=" * 60)
    # print("\nEnter a solution string with \\boxed{...} (or 'quit' to exit):")
    
    # while True:
    #     try:
    #         user_input = input("\nSolution> ").strip()
    #         if user_input.lower() in ['quit', 'exit', 'q']:
    #             break
    #         if not user_input:
    #             continue
            
    #         ground_truth = input("Ground truth> ").strip()
    #         if not ground_truth:
    #             print("  ⚠️  Ground truth cannot be empty")
    #             continue
            
    #         # Extract boxed content
    #         boxed_str = last_boxed_only_string(user_input)
    #         print(f"\n  Extracted boxed: {repr(boxed_str)}")
            
    #         if boxed_str:
    #             answer = remove_boxed(boxed_str)
    #             print(f"  Answer: {repr(answer)}")
    #             is_eq = is_equiv(answer, ground_truth)
    #             print(f"  Equivalent: {is_eq}")
            
    #         # Full compute_score
    #         gt_dict = {"target": ground_truth}
    #         result_train = compute_score(user_input, gt_dict, state="train")
    #         result_val = compute_score(user_input, gt_dict, state="val")
            
    #         print(f"  Train result: {result_train}")
    #         print(f"  Val result:   {result_val}")
            
    #     except KeyboardInterrupt:
    #         print("\n\nExiting...")
    #         break
    #     except Exception as e:
    #         print(f"  Error: {e}")
    #         import traceback
    #         traceback.print_exc()
