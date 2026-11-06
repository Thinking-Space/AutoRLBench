import ast
import os
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace


TEST_CASE_EXTRACTION_PROMPT = '''You are given:

1. An image that contains an example of the problem.
2. A Python function signature and docstring describing the task.
3. A candidate Python implementation that attempts to solve the problem.

Your task is to extract **one concrete test case from the image** and convert it into a Python `assert` statement to verify the correctness of the provided implementation.

Important:
- The image only provides an example instance of the problem.
- Carefully read the values, coordinates, numbers, or structures shown in the image.
- Convert the visual example into **explicit input arguments and the correct output**.
- Use these to construct a valid `assert` test.

Requirements:
- Use the function exactly as defined in the provided signature.
- The test must follow standard Python syntax.
- The format must be a single `assert` statement.

Output format:
Return only the Python assertion, for example:

`assert solution(input_example) == expected_output`'''


CODE_GENERATION_PROMPT = """You are solving a Python coding problem from an image example.

Requirements:
- Read the image example and the Python function signature carefully.
- Implement the function exactly as defined.
- Return a complete, runnable Python solution.
- Do not include tests.
- Return only one final Python code block in the format ```python ... ```.
"""


CODE_REGENERATION_PROMPT = """Your previous response did not contain a valid Python code block.

Requirements:
- Return a complete, runnable Python solution for the given task.
- Implement the function exactly as defined.
- Do not include tests.
- Return only one final Python code block in the format ```python ... ```.
"""


CODE_REPAIR_PROMPT = """Your previous code failed the extracted test case.

Fix the implementation using the image example, the task description, the failing code, and the error message.

Requirements:
- Keep the function signature exactly unchanged.
- Return a complete, runnable corrected Python solution.
- Do not include tests.
- Return only one final Python code block in the format ```python ... ```.
"""


class Code_Problem_Solver_tool:
    def __init__(
        self,
        tool,
        client,
        model,
        system_prompt,
        question,
        images,
        temperature,
        top_p,
        timeout,
        have_image=False,
        image_paths=None,
        dataset="",
    ):
        self.tool = tool
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.question = question
        self.images = images
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image
        self.image_paths = image_paths if image_paths else []
        self.dataset = dataset
        self.max_rounds = tool.tool_config.get("max_rounds", 3)
        self.exec_timeout = tool.tool_config.get("exec_timeout", min(timeout, 10) if timeout else 10)

    def _build_messages(self, prompt_text, include_images=True):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if include_images and self.images:
            content = []
            for image in self.images:
                content.append({"type": "image_url", "image_url": {"url": f"{image}"}})
            content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt_text})
        return messages

    def _call_model(self, prompt_text, include_images=True):
        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt_text, include_images=include_images),
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
        )
        return self._clean_output_content(outputs)

    def _extract_content(self, outputs):
        if hasattr(outputs, "choices") and outputs.choices:
            return outputs.choices[0].message.content or ""
        return ""

    def _remove_think_content(self, text):
        if not isinstance(text, str) or text == "":
            return text
        closing_matches = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
        if closing_matches:
            text = text[closing_matches[-1].end():]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def _clean_output_content(self, outputs):
        if hasattr(outputs, "choices") and outputs.choices:
            outputs.choices[0].message.content = self._remove_think_content(
                outputs.choices[0].message.content
            )
        return outputs

    def _accumulate_usage(self, usage_totals, outputs):
        usage = getattr(outputs, "usage", None)
        if usage is None:
            return
        usage_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        usage_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        usage_totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

    def _attach_usage_totals(self, outputs, usage_totals):
        if outputs is None or isinstance(outputs, str):
            return outputs

        usage = getattr(outputs, "usage", None)
        if usage is None:
            outputs.usage = SimpleNamespace(**usage_totals)
            return outputs

        usage.prompt_tokens = usage_totals["prompt_tokens"]
        usage.completion_tokens = usage_totals["completion_tokens"]
        usage.total_tokens = usage_totals["total_tokens"]
        return outputs

    def _extract_signature_block(self):
        question = (self.question or "").strip()
        if "def " in question:
            return question
        return "def solution(*args, **kwargs):\n    pass"

    def _build_stub_code(self):
        signature_block = self._extract_signature_block().rstrip()
        if re.search(r"\n\s+pass\s*$", signature_block):
            return signature_block
        return signature_block + "\n    pass"

    def _normalize_assertion(self, text):
        cleaned = self._remove_think_content(text or "").strip()
        cleaned = cleaned.replace("`", "").strip()
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("assert "):
                try:
                    ast.parse(line)
                    return line
                except Exception:
                    continue

        match = re.search(r"(assert\s+.+)", cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(1).strip().splitlines()[0].strip()
            try:
                ast.parse(candidate)
                return candidate
            except Exception:
                return ""
        return ""

    def _extract_code_block(self, text):
        cleaned = self._remove_think_content(text or "")
        start_idx = cleaned.lower().rfind("```python")
        if start_idx == -1:
            return ""
        content_start = start_idx + len("```python")
        end_idx = cleaned.find("```", content_start)
        if end_idx == -1:
            code = cleaned[content_start:].strip()
        else:
            code = cleaned[content_start:end_idx].strip()
        if not code:
            return ""
        try:
            ast.parse(code)
            return code
        except Exception:
            return ""

    def _run_code_with_assert(self, code, assert_stmt):
        script = f"{code.rstrip()}\n\n{assert_stmt.strip()}\n"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
                handle.write(script)
                temp_path = handle.name

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.exec_timeout,
                cwd="/tmp",
            )
            passed = result.returncode == 0
            error_text = (result.stderr or result.stdout or "").strip()
            return passed, error_text
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {self.exec_timeout} seconds."
        except Exception as exc:
            return False, str(exc)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _build_test_prompt(self, candidate_code):
        return (
            f"{TEST_CASE_EXTRACTION_PROMPT}\n\n"
            "Python function signature and docstring:\n"
            f"{self._extract_signature_block()}\n\n"
            "Candidate Python implementation:\n"
            f"```python\n{candidate_code}\n```"
        )

    def _build_generation_prompt(self):
        return (
            f"{CODE_GENERATION_PROMPT}\n\n"
            "Problem:\n"
            f"{self.question}"
        )

    def _build_regeneration_prompt(self, previous_output):
        return (
            f"{CODE_REGENERATION_PROMPT}\n\n"
            "Problem:\n"
            f"{self.question}\n\n"
            "Previous invalid response:\n"
            f"{previous_output}"
        )

    def _build_repair_prompt(self, bad_code, assert_stmt, error_text):
        return (
            f"{CODE_REPAIR_PROMPT}\n\n"
            "Problem:\n"
            f"{self.question}\n\n"
            "Failing test case:\n"
            f"{assert_stmt}\n\n"
            "Previous code:\n"
            f"```python\n{bad_code}\n```\n\n"
            "Error message:\n"
            f"{error_text}"
        )

    def generate(self, is_test=False):
        if self.client is None:
            print("Code_Problem_Solver: client is None")
            return ""

        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        test_assert = ""
        for extract_round in range(2):
            # import pdb; pdb.set_trace()
            test_outputs = self._call_model(self._build_test_prompt(self._build_stub_code()), include_images=True)
            self._accumulate_usage(usage_totals, test_outputs)
            test_assert = self._normalize_assertion(self._extract_content(test_outputs))
            if is_test:
                print(f"\n[Code_Problem_Solver][Test Extraction Round {extract_round + 1}]")
                print(test_assert or self._extract_content(test_outputs))
            if test_assert:
                break

        last_outputs = None
        last_code = ""
        last_error = ""

        for round_idx in range(self.max_rounds):
            if round_idx == 0:
                prompt_text = self._build_generation_prompt()
            elif not last_code:
                prompt_text = self._build_regeneration_prompt(
                    self._extract_content(last_outputs) if last_outputs is not None else ""
                )
            else:
                prompt_text = self._build_repair_prompt(last_code, test_assert, last_error)

            outputs = self._call_model(prompt_text, include_images=True)
            self._accumulate_usage(usage_totals, outputs)
            content = self._extract_content(outputs)
            code = self._extract_code_block(content)

            if is_test:
                print(f"\n[Code_Problem_Solver][Generation Round {round_idx + 1}]")
                print(content)

            last_outputs = outputs
            last_code = code

            if not code:
                last_error = "No valid Python code block found."
                continue

            if not test_assert:
                outputs.choices[0].message.content = f"```python\n{code}\n```"
                return self._attach_usage_totals(outputs, usage_totals)

            passed, error_text = self._run_code_with_assert(code, test_assert)
            if is_test:
                print(f"[Code_Problem_Solver][Round {round_idx + 1} Test]")
                print(test_assert)
                print("[Code_Problem_Solver][Round Result]")
                print("passed" if passed else error_text)

            if passed:
                outputs.choices[0].message.content = f"```python\n{code}\n```"
                return self._attach_usage_totals(outputs, usage_totals)

            last_error = error_text or "Unknown execution error."

        if last_outputs is not None and last_code:
            last_outputs.choices[0].message.content = f"```python\n{last_code}\n```"
            return self._attach_usage_totals(last_outputs, usage_totals)

        return self._attach_usage_totals(last_outputs, usage_totals) if last_outputs is not None else ""



    # sample_with_think = (
    #     "<think>draft</think>\n"
    #     "noise\n"
    #     "```python\n"
    #     "def bad():\n"
    #     "    return 0\n"
    #     "```\n"
    #     "<think>final reasoning</think>\n"
    #     "```python\n"
    #     "def add(a, b):\n"
    #     "    return a + b\n"
    #     "```"
    # )
    # assert tool._extract_code_block(sample_with_think) == "def add(a, b):\n    return a + b"

    # sample_with_prefix = (
    #     "hidden scratchpad\n"
    #     "</think>\n"
    #     "```python\n"
    #     "def add(a, b):\n"
    #     "    return a + b\n"
    #     "```"
    # )
    # assert tool._extract_code_block(sample_with_prefix) == "def add(a, b):\n    return a + b"

    # assert tool._normalize_assertion("```assert add(1, 2) == 3```") == "assert add(1, 2) == 3"

    # passed, error_text = tool._run_code_with_assert(
    #     "def add(a, b):\n    return a + b",
    #     "assert add(1, 2) == 3",
    # )
    # assert passed, error_text

    # failed, error_text = tool._run_code_with_assert(
    #     "def add(a, b):\n    return a - b",
    #     "assert add(1, 2) == 3",
    # )
    # assert not failed
    # assert error_text

    # print("Code_Problem_Solver smoke tests passed.")
