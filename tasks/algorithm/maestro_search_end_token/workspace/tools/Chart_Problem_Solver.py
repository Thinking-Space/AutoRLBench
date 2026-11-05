import base64
import json
import os
import re
from types import SimpleNamespace


GENERAL_CHART_INSTRUCTIONS = """## General Chart Instructions
1. If the question involves values corresponding to specific positions (e.g., lower, last, top), you must match the information with the positions in the chart image to perform reasoning.
2. If the question requires interpretation based on numerical magnitude, you should reason according to the numerical values in the information.
3. This task originally requires answering based only on the image, meaning all positions should be interpreted according to the chart image itself.
4. In most cases, the numerical values of the chart can be determined from the values on the x-axis or y-axis.
5. Note that predicted information can be utilized. The predicted columns and rows are likely to correspond to the actual columns and rows in the chart, which helps determine the positions of rows and columns in the chart.
"""

CHART_R1_MAX_TOKENS = 2048


BAR_CHART_INSTRUCTIONS = """## Bar Chart Instructions
1. First, bars with the same color represent the same column. Therefore, distinguishing colors and identifying the corresponding columns is crucial (usually shown in the legend around the main chart).
2. Second, determine the positions of the rows. For vertical bar charts, rows are usually labeled at the bottom of the main chart; for horizontal bar charts, rows are labeled on the left or right side of the main chart.
3. Then, combine the color of the nearest bar with the labeled row to determine the row and column corresponding to that bar in the information.
4. Next, locate the value corresponding to each row and column. If values are labeled on the bars, refer to them; otherwise, compare the sizes of the bars to determine the values.
5. For vertical bar charts, the value of a bar corresponds to the y-axis value at the end of the bar. Similarly, for horizontal bar charts, the value corresponds to the x-axis value at the end of the bar.
"""


LINE_CHART_INSTRUCTIONS = """## Line Chart Instructions
1. In line charts, the x-axis at the bottom mainly represents rows, and each colored line represents a column.
2. The legend is usually located inside the main chart and indicates which column corresponds to each line color. If the legend is missing or placed separately, text labels corresponding to the line colors are likely to indicate columns (if there are no colors, the text labels at the left or right ends of the lines likely correspond to columns).
3. The point where a line passes through the same x-coordinate as each x-axis value represents the actual value (i.e., the x-axis corresponds to rows, the line color corresponds to columns, and the point represents the value).
4. If there are annotations near the points on the line, those annotations most likely represent the values of the points.
5. If there are no annotations near the points, the values can be determined from the y-axis values corresponding to the y-coordinates of the points.
6. In line charts, understanding the trend of the lines is crucial. Lines may show decreasing, increasing, or stable trends. When multiple lines intersect, it is important to identify which column each line represents based on color.
"""


PIE_CHART_INSTRUCTIONS = """## Pie Chart Instructions
1. In pie charts, it is important to determine which color corresponds to which row.
2. Each sector has a color, and the corresponding row may be indicated by text inside or near the sector. If not nearby, it may be identified through the legend or through lines/markers connecting to the corresponding text.
3. In pie charts, values are usually labeled on each sector of the pie chart.
"""


class Chart_Problem_Solver_tool:
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

        self.ocr_results_text = self._get_ocr_results_for_images()
        self.messages = self._build_preview_messages()

    def _build_preview_messages(self):
        preview_text = (
            f"Question:\n{self.question}\n\n"
            "Workflow summary:\n"
            "1. Read OCR result from JSON.\n"
            "2. Model call for chart type classification.\n"
            "3. Model call for chart-specific answering.\n"
        )
        return [{"role": "user", "content": preview_text}]

    def _get_ocr_results_file(self):
        return (
            "results/tool_results/ocr.json"
            if self.dataset == ""
            else "results/tool_results/ocr_eval.json"
        )

    def _load_ocr_result_data(self):
        result_file = self._get_ocr_results_file()
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取ocr结果文件 {result_file} 时出错: {e}")
            return {}

    def _get_ocr_results_for_images(self):
        if not self.image_paths:
            return ""

        ocr_data = self._load_ocr_result_data()
        ocr_results = []

        for i, image_path in enumerate(self.image_paths):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_key = image_name if self.dataset == "" else f"{self.dataset}_{image_name}"

            if image_key in ocr_data and ocr_data[image_key]:
                ocr_results.append(f"The ocr result of image{i+1} is: {ocr_data[image_key]}")
            else:
                ocr_results.append(f"The ocr result of image{i+1} is: Not found")

        return "\n\n".join(ocr_results) if ocr_results else ""

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
        request_kwargs = {
            "model": self.model,
            "messages": self._build_messages(prompt_text, include_images=include_images),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }
        if "chart-r1" in str(self.model).lower():
            request_kwargs["max_tokens"] = CHART_R1_MAX_TOKENS
        return self.client.chat.completions.create(**request_kwargs)

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

    def _normalize_chart_type(self, chart_type_response):
        chart_type_lower = chart_type_response.lower()
        if "pie" in chart_type_lower:
            return "pie chart"
        if "line" in chart_type_lower:
            return "line chart"
        if "bar" in chart_type_lower:
            return "bar chart"
        return "bar chart"

    def _build_step2_prompt(self):
        ocr_text = self.ocr_results_text if self.ocr_results_text else "No OCR result found."
        return (
            "You are classifying the type of chart in the image.\n\n"
            f"Question:\n{self.question}\n\n"
            "OCR evidence read from file:\n"
            f"{ocr_text}\n\n"
            "Decide whether the chart is a bar chart, line chart, or pie chart.\n"
            "Use the image as the primary evidence and OCR as supporting evidence.\n"
            "If the OCR is noisy, still infer the chart type from visual structure.\n"
            "Respond with exactly one of:\n"
            "bar chart\n"
            "line chart\n"
            "pie chart\n"
        )

    def _get_chart_type_instructions(self, chart_type):
        if chart_type == "line chart":
            return LINE_CHART_INSTRUCTIONS
        if chart_type == "pie chart":
            return PIE_CHART_INSTRUCTIONS
        return BAR_CHART_INSTRUCTIONS

    def _build_step3_prompt(self, chart_type):
        ocr_text = self.ocr_results_text if self.ocr_results_text else "No OCR result found."
        chart_specific_instructions = self._get_chart_type_instructions(chart_type)
        return (
            "You are solving a chart question from an image.\n\n"
            f"Question:\n{self.question}\n\n"
            "Step 1 OCR evidence read from file:\n"
            f"{ocr_text}\n\n"
            "Step 2 chart type decision:\n"
            f"{chart_type}\n\n"
            "# Chart Operation Instructions\n\n"
            f"{GENERAL_CHART_INSTRUCTIONS}\n\n"
            f"{chart_specific_instructions}\n\n"
            "Answer the question based on the image itself, the OCR evidence, and the chart-type-specific rules above.\n"
            "If the question depends on positions, align those positions with the actual chart layout in the image.\n"
            "If the question depends on values, infer them from the axes, labels, annotations, or visual magnitudes.\n"
            "Give a concise reasoning process and then provide the final answer on the last line."
        )

    def generate(self, is_test=False):
        try:
            usage_totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            if self.client is None:
                print("Chart_Problem_Solver: client is None")
                return ""

            step2_outputs = self._call_model(self._build_step2_prompt(), include_images=True)
            self._accumulate_usage(usage_totals, step2_outputs)
            step2_response = self._remove_think_content(self._extract_content(step2_outputs))
            chart_type = self._normalize_chart_type(step2_response)
            if is_test:
                print("\n[Chart_Problem_Solver][Step 1 OCR Result]")
                print(self.ocr_results_text)
                print("\n[Chart_Problem_Solver][Step 2 Output]")
                print(step2_response)
                print("\n[Chart_Problem_Solver][Normalized Chart Type]")
                print(chart_type)

            step3_outputs = self._call_model(
                self._build_step3_prompt(chart_type),
                include_images=True,
            )
            self._accumulate_usage(usage_totals, step3_outputs)
            # step3_outputs = self._clean_output_content(step3_outputs)
            step3_outputs = self._attach_usage_totals(step3_outputs, usage_totals)
            if is_test:
                print("\n[Chart_Problem_Solver][Step 3 Output]")
                print(self._extract_content(step3_outputs))
            return step3_outputs
        except Exception as e:
            print(f"生成响应时出错(Chart_Problem_Solver): {e}")
            return ""


