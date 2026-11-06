import ast
import os
import re
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir in sys.path:
        sys.path.remove(current_dir)

import tempfile
import subprocess


def _get_test_script(ground_truth):
    if isinstance(ground_truth, list):
        return str(ground_truth[0]).strip() if ground_truth else ""
    if isinstance(ground_truth, dict):
        for key in ("answer", "test", "tests", "ground_truth"):
            if key in ground_truth and ground_truth[key] is not None:
                return str(ground_truth[key]).strip()
        return ""
    return str(ground_truth).strip() if ground_truth is not None else ""
def _is_valid_python(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def extract_code(solution_str: str) -> str:
    source = solution_str or ""
    think_end_idx = source.rfind("</think>")
    if think_end_idx != -1:
        source = source[think_end_idx + len("</think>"):]

    start_idx = source.lower().rfind("```python")
    if start_idx != -1:
        content_start = start_idx + len("```python")
        end_idx = source.find("```", content_start)
        if end_idx != -1:
            code = source[content_start:end_idx].strip()
            if _is_valid_python(code):
                return code

    return solution_str


def _run_test_script(code: str, test_script: str, timeout: int = 10) -> tuple[bool, str]:
    if not code:
        return False, "No executable Python code found."
    if not test_script:
        return False, "No test script found in ground truth."

    script_content = f"{code.rstrip()}\n\n{test_script.strip()}\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(script_content)
            temp_path = handle.name

        completed = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
        )
        error_output = (completed.stderr or completed.stdout or "").strip()
        return completed.returncode == 0, error_output
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s."
    except Exception as exc:
        return False, str(exc)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def compute_score(
    solution_str,
    ground_truth,
    format_score=0.0,
    score=1.0,
    cost_coe=0.0,
    api_cost=0.0,
    state="train",
    reward_metric="f1",
    return_answer=False,
    timeout=10,
):
    del score, reward_metric
    extracted_code = extract_code(solution_str)
    test_script = _get_test_script(ground_truth)
    passed, _ = _run_test_script(extracted_code, test_script, timeout=timeout)
    # import pdb; pdb.set_trace()
    metric = 1.0 if passed else 0.0

    if state == "train":
        if format_score == -1.0:
            score_info = (metric, api_cost, format_score)
        elif metric == 0:
            score_info = (metric, api_cost, metric + format_score)
        else:
            score_info = (metric, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)
    else:
        metric_em = metric
        metric_f1 = metric
        if format_score == -1.0:
            score_info = (metric_em, metric_f1, api_cost, format_score)
        elif metric == 0:
            score_info = (metric_em, metric_f1, api_cost, metric + format_score)
        else:
            score_info = (metric_em, metric_f1, api_cost, (metric + format_score) * (1.0 - cost_coe) + api_cost * cost_coe)

    if return_answer:
        return score_info, extracted_code
    return score_info


def compute_score_humaneval_v(solution_str, ground_truth, format_score=0.0, **kwargs):
    return compute_score(solution_str, ground_truth, format_score=format_score, **kwargs)


if __name__ == "__main__":
    sample_test = """
def test():\n    test_cases = [\n        ((2, 0), (2, 1), 'A'),\n        ((0, 0), (4, 0), 'C'),\n        ((1, 0), (2, 0), 'B'),\n        ((0, 0), (0, 0), 'A'),\n        ((3, 10), (4, 4), 'C'),\n        ((5, 6), (7, 4), 'B'),\n        ((9, 9), (9, 10), 'B'),\n        ((999, 1002), (1000, 1000), 'C'),\n        ((0, 0), (0, 1), 'B'),\n        ((0, 0), (1, 0), 'B')\n    ]\n    for point1, point2, expected_output in test_cases:\n        try:\n            output = solution(point1, point2)\n        except Exception as e:\n            error_message = (\n                f\"An exception occurred while running the test case:\\n\"\n                f\"`point1`: {point1}\\n\"\n                f\"`point2`: {point2}\\n\\n\"\n                f\"Exception: '{str(e)}'\"\n            )\n            assert False, error_message\n\n        if output != expected_output:\n            error_message = (\n                f\"The following Test Case failed:\\n\"\n                f\"`point1`: {point1}\\n\" \n                f\"`point2`: {point2}\\n\\n\"\n                f\"Actual Output: {output}\\n\"\n                f\"Expected Output: {expected_output}\"\n            )\n            assert False, error_message\n\ntest()
""".strip()

    passing_response = """
So, let's try to figure out this problem. First, let's understand the image. The image shows layers (L0 to L4) along the y-axis, and x-axis has L1 to L4 (but Layer 0 is at x=0, y=0 maybe? Wait, the x-axis labels are Layer 0 (at x=0), then L1, L2, L3, L4. The y-axis is labeled L4 at the top, down to Layer 0 at the bottom. So each point's layer is determined by the y-coordinate. Let's check:\n\nFor a point (x, y), the layer is L(y). Because L4 is the top y, L3 is next, down to Layer 0 at y=0. So the layer of a point is its y-coordinate. For example, a point on L4 has y=4, L3 has y=3, etc. Then, the x-coordinate: the x-axis has positions. Let's see the x labels: Layer 0 is at x=0, then L1 is x=1, L2 x=2, L3 x=3, L4 x=4. So a point's x is the horizontal position, and y is the vertical layer.\n\nNow, the labels A, B, C:\n\n- \"A\": Same Layer. So two points are on the same y (same layer). The orange arrows show two points on L4 (y=4) connected by A. So if point1 and point2 have the same y-coordinate, return 'A'.\n\n- \"B\": Adjacent Layer. The green arrow. Let's check the points. The green arrow goes from L3 (y=3) to L4 (y=4)? Wait, no. Let's look at the green arrow: the start is a point on L3 (the dashed circle on L3, L2 column? Wait, the green arrow's start is the dashed circle on L3, L2 x? Wait, the green arrow is from the dashed circle on L3 (y=3) at x=2 (since L2 is x=2) to the solid circle on L4 (y=4) at x=4. Wait, no, the green arrow's end is the solid circle on L4, x=4. The start is the dashed circle on L3, x=2. So the y difference is 1 (L4 - L3 = 1), and x difference? Let's check the layers. Adjacent layer means the y-coordinates differ by 1. Because adjacent layers are L3 and L4 (difference 1), L2 and L3, etc. So if the absolute difference between y1 and y2 is 1, then it's adjacent layer (B). Let's check the green arrow: the two points are on L3 and L4, so y difference is 1. So B is when |y1 - y2| == 1.\n\n- \"C\": Non-Adjacent Layer. The red arrow. Let's check the red arrow: it goes from a point on L1 (y=1) at x=2 (the dashed circle on L1, x=2) to a point on L3 (y=3) at x=4 (the dashed circle on L2, x=4? Wait, no. The red arrow's start is the dashed circle on L1, x=2 (the circled point on L1, x=2) and the end is the dashed circle on L2, x=4 (the circled point on L2, x=4). Wait, the y of the start is L1 (y=1), the y of the end is L2 (y=2)? No, the red arrow's label is \"C: Non-Adjacent Layer\". Let's check the y difference. The start point is on L1 (y=1), the end point is on L3 (y=3)? Wait, no. Let's look at the red arrow: the start is the circled point on L1 (x=2, y=1) and the end is the circled point on L2 (x=4, y=2)? No, the red arrow's end is the dashed circle on L2, x=4 (the point on L2, x=4 is a dashed circle? Wait, the image has:\n\nLooking at the red arrow: the dashed red line goes from the circled point on L1 (x=2, y=1) to the dashed circle on L2 (x=4, y=2)? No, the red arrow's end is the dashed circle on L2, x=4? Wait, the label \"C\" is pointing to a red arrow that goes from a point on L1 (y=1) to a point on L3 (y=3)? Let's check the y difference. If the y difference is more than 1, then it's non-adjacent. For example, L1 (y=1) and L3 (y=3) have a difference of 2, which is non-adjacent. Let's check the red arrow: the start is on L1 (y=1) and the end is on L3 (y=3)? Let's see the red arrow's path: the red dashed line starts at the circled point on L1 (x=2, y=1) and goes to the dashed circle on L2 (x=4, y=2)? No, the red arrow's end is the dashed circle on L2, x=4? Wait, the image's red arrow is labeled \"C: Non-Adjacent Layer\". Let's check the y of the two points in the red arrow. Let's take the start point: the circled point on L1 (x=2, y=1) — that's a point on layer L1 (y=1). The end point of the red arrow is the dashed circle on L2 (x=4, y=2)? No, the red arrow's end is the dashed circle on L2, x=4? Wait, no, the red arrow's end is the dashed circle on L2, x=4? Let me check the image again.\n\nAlternatively, maybe the x-coordinate is not the key, but the layer (y) and the x difference. Wait, no. Let's think about the layers. The image's x-axis is labeled Layer 0 (at x=0), L1 (x=1), L2 (x=2), L3 (x=3), L4 (x=4). The y-axis is labeled L4 (top, y=4), L3 (y=3), L2 (y=2), L1 (y=1), Layer 0 (y=0, bottom). So each point (x, y) is in layer L(y). So the layer of a point is determined by its y-coordinate. Now, the three cases:\n\n- A: Same layer → y1 == y2 → return 'A'\n- B: Adjacent layer → |y1 - y2| == 1 → return 'B'\n- C: Non-adjacent layer → |y1 - y2| > 1 → return 'C'\n\nWait, let's check the image's labels. The \"B\" is labeled \"Adjacent Layer\". Let's check the green arrow. The green arrow connects a point on L3 (y=3) to a point on L4 (y=4). The y difference is 1, so that's adjacent. The \"C\" is \"Non-Adjacent Layer\". Let's check the red arrow. The red arrow connects a point on L1 (y=1) to a point on L3 (y=3). The y difference is 2, which is more than 1, so non-adjacent. The \"A\" is same layer: two points on L4 (y=4), so y difference 0. That matches.\n\nSo the logic is based on the y-coordinates of the two points. Let's confirm:\n\nFor point1 (x1, y1) and point2 (x2, y2):\n\n- If y1 == y2 → 'A'\n- Else if abs(y1 - y2) == 1 → 'B'\n- Else → 'C'\n\nBecause the x-coordinate doesn't matter for the layer relationship. The image's x is just the horizontal position, but the layer is determined by y. Let's check the image's \"A\" case: two points on L4 (y=4) — same y, so A. \"B\" case: L3 (y=3) and L4 (y=4) — y difference 1, so B. \"C\" case: L1 (y=1) and L3 (y=3) — y difference 2, so C. That fits.\n\nSo the function should check the y-coordinates of the two points. Let's code that.\n\nThe function signature is given: def solution(point1: Tuple[int, int], point2: Tuple[int, int]) -> str:\n\nSo point1 is (x1, y1), point2 is (x2, y2). We need to extract y1 and y2.\n\nThen:\n\ny1 = point1[1]\ny2 = point2[1]\n\nif y1 == y2:\n    return 'A'\nelif abs(y1 - y2) == 1:\n    return 'B'\nelse:\n    return 'C'\n\nLet's test with the image's examples.\n\nTest case 1: \"A\" case. Two points on L4 (y=4). So point1 = (1,4), point2 = (2,4). y1 and y2 are 4. So return 'A'.\n\nTest case 2: \"B\" case. Point on L3 (y=3) and L4 (y=4). point1 = (2,3), point2 = (4,4). y difference is 1. Return 'B'.\n\nTest case 3: \"C\" case. Point on L1 (y=1) and L3 (y=3). point1 = (2,1), point2 = (4,3). y difference is 2. Return 'C'.\n\nYes, that matches the image's labels.\n\nSo the code should implement this logic.\n</think>\n```python\nfrom typing import Tuple\n\ndef solution(point1: Tuple[int, int], point2: Tuple[int, int]) -> str:\n    \"\"\"\n    Determine the relationship between two points based on their layers.\n\n    Parameters:\n        point1 (Tuple[int, int]): The coordinates of the first point, where both x and y are non-negative integers.\n        point2 (Tuple[int, int]): The coordinates of the second point, where both x and y are non-negative integers.\n\n    Returns:\n        str: Return 'A', 'B', or 'C' based on layer relationship.\n    \"\"\"\n    y1 = point1[1]\n    y2 = point2[1]\n    if y1 == y2:\n        return \"A\"\n    elif abs(y1 - y2) == 1:\n        return \"B\"\n    else:\n        return \"C\"\n```

"""
    no_code_response = "<answer>最终答案是 42</answer>"

    extracted_code = extract_code(passing_response)
    print("[test] extracted code:")
    print(extracted_code)
    # assert extracted_code.startswith("def solution")

    passing_score = compute_score(passing_response, sample_test)
    print("[test] passing score:", passing_score)
    # assert passing_score == (1.0, 0.0, 1.0)

