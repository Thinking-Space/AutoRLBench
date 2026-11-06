import base64
import json
import os
import re
from types import SimpleNamespace


EXTRACTION_PROMPT = """# Extract ALL visual and text information in this structured format:

## 1. POINTS
- List each labeled point and its type (vertex, intersection, center, etc.)
- Format: "Label: [type] at approximate coordinates (x,y relative)"

## 2. LINES & SEGMENTS
For each line/segment:
- Endpoints
- Type (solid, dashed, dotted)
- Special markings (arrow for ray, tick marks for equal length)
- Labels if present

## 3. ANGLES
For each marked angle:
- Vertex point
- Arms (two points forming the angle)
- Measurement if given
- Right angle/square symbol if present

## 4. CIRCLES/ARCS
- Center point
- Radius/diameter if labeled
- Arc endpoints if partial

## 5. ANNOTATIONS
- Length measurements (with units)
- Angle measurements (with degree symbol)
- Ratio markings (single/double tick marks)
- Textual labels not attached to points

## 6. RELATIONSHIPS
- Parallel lines (mark with //)
- Perpendicular lines (mark with ⊥)
- Tangency points
- Collinearity (points on same line)

Output this as a structured list with clear headings.
"""


STEP4_PROMPT = """Compare your own visual understanding with the outputs from imagecaption and OCR, then determine the most accurate description.

Requirements:
- Start from your own visual extraction as the primary evidence.
- Use imagecaption as global descriptive support.
- Use OCR as textual evidence for labels, numbers, and symbols.
- Explicitly point out conflicts or ambiguities.
- Resolve conflicts by choosing the most visually and logically consistent interpretation.
- Output a concise but precise consolidated description of the geometry problem setup.
"""


STEP5_PROMPT = """Begin step-by-step reasoning with self-explanation to solve the problem.

Requirements:
- Restate the target quantity or statement to prove.
- List the key known facts you will use.
- Identify the geometric theorems, angle/length relations, symmetry, parallel-line rules, triangle facts, circle facts, or coordinate facts that apply.
- Derive the result step by step without skipping key logical steps.
- If there are multiple interpretations, choose the best-supported one and say so briefly.
- End this step with a provisional answer, but do not do the final verification yet.
"""


FINAL_CHECK_PROMPT = """# Check the entire solution for consistency:

## 1. DIMENSIONAL ANALYSIS
- Do units match throughout?
- Are angle measures in consistent units?
- Do ratios make sense?

## 2. EXTREME CASE TESTING
- Does solution hold for special configurations?
- What if a measurement approaches zero/infinity?
- Boundary condition check

## 3. CROSS-METHOD VERIFICATION
- Can result be obtained via alternative method?
- Quick mental estimation vs. calculated result
- Symmetry considerations

## 4. PLAUSIBILITY CHECK
Is magnitude reasonable?
"""


STEP6_PROMPT = """Review the entire reasoning process and check the final solution for consistency and correctness.

Requirements:
- Apply the consistency checklist below carefully.
- If the previous reasoning contains an error, correct it.
- If the previous reasoning is sound, confirm it briefly.
- The final line must be the final answer only, stated clearly in the format the question asks for.

Consistency checklist:
""" + FINAL_CHECK_PROMPT


class Geometric_Problem_Solver_tool:
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
        image_paths=[],
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

        self.caption_results_text = self._get_caption_results_for_images()
        self.ocr_results_text = self._get_ocr_results_for_images()
        self.messages = self._build_preview_messages()

    def _build_preview_messages(self):
        preview_text = (
            f"Question:\n{self.question}\n\n"
            "Workflow summary:\n"
            "1. Model call for structured extraction.\n"
            "2. Read imagecaption result from JSON.\n"
            "3. Read OCR result from JSON.\n"
            "4. Model call for evidence fusion.\n"
            "5. Model call for step-by-step solving.\n"
            "6. Model call for final verification and final answer.\n"
        )
        return [{"role": "user", "content": preview_text}]

    def _get_results_file(self, result_type):
        if result_type == "caption":
            return (
                "results/tool_results/caption.json"
                if self.dataset == ""
                else "results/tool_results/caption_eval.json"
            )
        if result_type == "ocr":
            return (
                "results/tool_results/ocr.json"
                if self.dataset == ""
                else "results/tool_results/ocr_eval.json"
            )
        raise ValueError(f"Unsupported result type: {result_type}")

    def _load_result_data(self, result_type):
        result_file = self._get_results_file(result_type)
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取{result_type}结果文件 {result_file} 时出错: {e}")
            return {}

    def _format_result_for_images(self, result_type):
        if not self.image_paths:
            return ""

        result_data = self._load_result_data(result_type)
        formatted_results = []
        label = "caption" if result_type == "caption" else "ocr"

        for i, image_path in enumerate(self.image_paths):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_key = image_name if self.dataset == "" else f"{self.dataset}_{image_name}"

            if image_key in result_data and result_data[image_key]:
                formatted_results.append(
                    f"The {label} result of image{i+1} is: {result_data[image_key]}"
                )
            else:
                if result_type == "caption":
                    formatted_results.append(f"The {label} result of image{i+1} is: ")
                else:
                    formatted_results.append(f"The {label} result of image{i+1} is: Not found")

        return "\n\n".join(formatted_results) if formatted_results else ""

    def _get_caption_results_for_images(self):
        return self._format_result_for_images("caption")

    def _get_ocr_results_for_images(self):
        return self._format_result_for_images("ocr")

    def _build_messages(self, prompt_text, include_images=True):
        if self.system_prompt == "":
            messages = []
        else:
            messages = [{"role": "system", "content": self.system_prompt}]

        if include_images and self.images:
            content = []
            for image in self.images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image}"
                        },
                    }
                )
            content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt_text})

        return messages

    def _call_model(self, prompt_text, include_images=True):
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt_text, include_images=include_images),
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
        )

    def _extract_content(self, outputs):
        if hasattr(outputs, "choices") and outputs.choices:
            return outputs.choices[0].message.content
        return ""

    def _remove_think_content(self, text):
        if not isinstance(text, str) or text == "":
            return text
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def _clean_output_content(self, outputs):
        if hasattr(outputs, "choices") and outputs.choices:
            content = outputs.choices[0].message.content
            outputs.choices[0].message.content = self._remove_think_content(content)
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

    def _build_step1_prompt(self):
        return (
            "You are an expert geometry diagram analyst.\n\n"
            f"Question:\n{self.question}\n\n"
            "Step 1. Inspect the image carefully by yourself and extract all visual and textual information.\n"
            "Use the following prompt exactly:\n"
            f"{EXTRACTION_PROMPT}\n"
        )

    def _build_step4_prompt(self, step1_response):
        caption_text = self.caption_results_text if self.caption_results_text else "No imagecaption result found."
        ocr_text = self.ocr_results_text if self.ocr_results_text else "No OCR result found."

        return (
            "You are integrating evidence for a geometry problem.\n\n"
            f"Question:\n{self.question}\n\n"
            "Your own visual extraction from Step 1:\n"
            f"{step1_response}\n\n"
            "Step 2 imagecaption evidence read from file:\n"
            f"{caption_text}\n\n"
            "Step 3 OCR evidence read from file:\n"
            f"{ocr_text}\n\n"
            "Step 4 instruction:\n"
            f"{STEP4_PROMPT}\n"
        )

    def _build_step5_prompt(self, step1_response, step4_response):
        caption_text = self.caption_results_text if self.caption_results_text else "No imagecaption result found."
        ocr_text = self.ocr_results_text if self.ocr_results_text else "No OCR result found."

        return (
            "You are solving a geometry problem.\n\n"
            f"Question:\n{self.question}\n\n"
            "Structured extraction from Step 1:\n"
            f"{step1_response}\n\n"
            "Imagecaption evidence from Step 2:\n"
            f"{caption_text}\n\n"
            "OCR evidence from Step 3:\n"
            f"{ocr_text}\n\n"
            "Consolidated description from Step 4:\n"
            f"{step4_response}\n\n"
            "Step 5 instruction:\n"
            f"{STEP5_PROMPT}\n"
        )

    def _build_step6_prompt(self, step1_response, step4_response, step5_response):
        caption_text = self.caption_results_text if self.caption_results_text else "No imagecaption result found."
        ocr_text = self.ocr_results_text if self.ocr_results_text else "No OCR result found."

        return (
            "You are performing the final verification for a geometry problem.\n\n"
            f"Question:\n{self.question}\n\n"
            "Step 1 structured extraction:\n"
            f"{step1_response}\n\n"
            "Step 2 imagecaption evidence:\n"
            f"{caption_text}\n\n"
            "Step 3 OCR evidence:\n"
            f"{ocr_text}\n\n"
            "Step 4 consolidated description:\n"
            f"{step4_response}\n\n"
            "Step 5 step-by-step solution:\n"
            f"{step5_response}\n\n"
            "Step 6 instruction:\n"
            f"{STEP6_PROMPT}\n"
        )

    def generate(self, is_test=False):
        try:
            usage_totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            if self.client is None:
                print("Geometric_Problem_Solver: client is None")
                return ""

            step1_outputs = self._call_model(self._build_step1_prompt(), include_images=True)
            self._accumulate_usage(usage_totals, step1_outputs)
            step1_response = self._remove_think_content(self._extract_content(step1_outputs))
            if is_test:
                print("\n[Geometric_Problem_Solver][Step 1 Output]")
                print(step1_response)

            step4_outputs = self._call_model(
                self._build_step4_prompt(step1_response),
                include_images=True,
            )
            self._accumulate_usage(usage_totals, step4_outputs)
            step4_response = self._remove_think_content(self._extract_content(step4_outputs))
            if is_test:
                print("\n[Geometric_Problem_Solver][Step 4 Output]")
                print(step4_response)

            step5_outputs = self._call_model(
                self._build_step5_prompt(step1_response, step4_response),
                include_images=True,
            )
            self._accumulate_usage(usage_totals, step5_outputs)
            step5_response = self._remove_think_content(self._extract_content(step5_outputs))
            if is_test:
                print("\n[Geometric_Problem_Solver][Step 5 Output]")
                print(step5_response)

            step6_outputs = self._call_model(
                self._build_step6_prompt(step1_response, step4_response, step5_response),
                include_images=True,
            )
            self._accumulate_usage(usage_totals, step6_outputs)
            # step6_outputs = self._clean_output_content(step6_outputs)
            step6_outputs = self._attach_usage_totals(step6_outputs, usage_totals)
            if is_test:
                print("\n[Geometric_Problem_Solver][Step 6 Output]")
                print(self._extract_content(step6_outputs))
            return step6_outputs
        except Exception as e:
            print(f"生成响应时出错(Geometric_Problem_Solver): {e}")
            return ""


