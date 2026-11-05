import base64
import io
import json
import os
import re
import time
from ast import literal_eval
from types import SimpleNamespace
from tools.service_config import get_service

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from tools.api_client import OpenAI
except ImportError:
    OpenAI = None


DEEPEYES_SYSTEM_PROMPT = """You are a helpful assistant.
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}},"required":["bbox_2d"]}}}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}
</tool_call>"""


FINAL_SCIENCE_PROMPT = """You are solving a science question from an image.

Use the following evidence:
1. The original image.
2. The zoomed-in crop if provided.
3. The image caption result, which gives an overall description.
4. The OCR result, which gives text found in the image.
5. The original question.

Requirements:
- Use the image itself as the primary source of truth.
- Use the image caption as global context.
- Use OCR for text, numbers, labels, units, and symbols.
- Use the zoomed crop to inspect the most relevant local region.
- If any source conflicts with the image, trust the image.
- Give a concise reasoning process.
- The final line must be the final answer only, stated in the format required by the question.
"""


class Science_Problem_Solver_tool:
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

        service = get_service('deepeyes')
        self.deepeyes_base_url = service.base_url
        self.deepeyes_model_id = service.served_model_name
        self.deepeyes_client = None
        if OpenAI is not None:
            try:
                self.deepeyes_client = OpenAI(base_url=self.deepeyes_base_url, api_key="EMPTY")
            except Exception as e:
                print(f"初始化 DeepEyes-7B 客户端失败: {e}")

    def _build_preview_messages(self):
        preview_text = (
            f"Question:\n{self.question}\n\n"
            "Workflow summary:\n"
            "1. Read imagecaption result from JSON.\n"
            "2. Read OCR result from JSON.\n"
            "3. Use DeepEyes-7B to decide whether to zoom and where to crop.\n"
            "4. Answer with the original image, cropped image, caption result, OCR result, and question.\n"
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

    def _build_messages(self, prompt_text, extra_images=None, include_images=True):
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
                        "image_url": {"url": f"{image}"},
                    }
                )
            for image in extra_images or []:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{image}"},
                    }
                )
            content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt_text})

        return messages

    def _call_model(self, prompt_text, extra_images=None, include_images=True):
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt_text, extra_images=extra_images, include_images=include_images),
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

    def _get_deepeyes_decision(self, img_url):
        if self.deepeyes_client is None:
            print("DeepEyes-7B client is unavailable")
            return "", None

        user_prompt = (
            f"\nQuestion: {self.question}\n"
            "Think first, call **image_zoom_in_tool** if a local region must be inspected to answer the science question accurately, "
            "then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>"
        )

        messages = [
            {"role": "system", "content": DEEPEYES_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        try:
            response = self.deepeyes_client.chat.completions.create(
                model=self.deepeyes_model_id,
                messages=messages,
                temperature=0.0,
                stop=["<|im_end|>\n", "</answer>", "</tool_call>"],
                timeout=self.timeout,
            )
            return response.choices[0].message.content, response
        except Exception as e:
            print(f"DeepEyes-7B 调用失败(Science): {e}")
            return "", None

    def _crop_image(self, image_path, deepeyes_output, save_dir=None):
        try:
            if Image is None:
                print("裁剪图像失败: Pillow is not installed")
                return None, None
            call_str = deepeyes_output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            try:
                action = json.loads(call_str.replace("'", '"'))
            except Exception:
                action = literal_eval(call_str)

            bbox = action["arguments"]["bbox_2d"]

            with Image.open(image_path) as img:
                crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                saved_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    timestamp = int(time.time() * 1000)
                    saved_path = os.path.join(save_dir, f"{image_name}_{timestamp}_science_crop.jpg")
                    crop.save(saved_path, format="JPEG")

                buffered = io.BytesIO()
                crop.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_str}", saved_path
        except Exception as e:
            print(f"裁剪图像失败: {e}")
            return None, None

    def _build_final_prompt(self, has_crop):
        caption_text = self.caption_results_text if self.caption_results_text else "No imagecaption result found."
        ocr_text = self.ocr_results_text if self.ocr_results_text else "No OCR result found."

        crop_text = (
            "A zoomed-in crop of the most relevant local region is also provided."
            if has_crop
            else "No zoomed-in crop was provided because no extra magnification was judged necessary."
        )

        return (
            f"{FINAL_SCIENCE_PROMPT}\n\n"
            f"Question:\n{self.question}\n\n"
            "Imagecaption evidence:\n"
            f"{caption_text}\n\n"
            "OCR evidence:\n"
            f"{ocr_text}\n\n"
            f"{crop_text}\n"
        )

    def generate(self, is_test=False):
        try:
            usage_totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            if self.client is None:
                print("Science_Problem_Solver: client is None")
                return ""
            if not self.images:
                print("Science_Problem_Solver: no images provided")
                return ""

            if is_test:
                print("\n[Science_Problem_Solver][Step 1 Caption Result]")
                print(self.caption_results_text)
                print("\n[Science_Problem_Solver][Step 2 OCR Result]")
                print(self.ocr_results_text)

            main_img_url = self.images[0]
            main_img_path = self.image_paths[0] if self.image_paths else None

            deepeyes_res, deepeyes_outputs = self._get_deepeyes_decision(main_img_url)
            self._accumulate_usage(usage_totals, deepeyes_outputs)
            if is_test:
                print("\n[Science_Problem_Solver][Step 3 DeepEyes Output]")
                print(deepeyes_res)

            crop_url = None
            if "<tool_call>" in deepeyes_res and main_img_path:
                save_dir = "tools/test_zoom" if is_test else None
                crop_url, saved_path = self._crop_image(main_img_path, deepeyes_res, save_dir=save_dir)
                if is_test and saved_path:
                    print(f"[Science_Problem_Solver] Cropped image saved to: {saved_path}")

            final_outputs = self._call_model(
                self._build_final_prompt(has_crop=bool(crop_url)),
                extra_images=[crop_url] if crop_url else None,
                include_images=True,
            )
            self._accumulate_usage(usage_totals, final_outputs)
            final_outputs = self._clean_output_content(final_outputs)
            final_outputs = self._attach_usage_totals(final_outputs, usage_totals)
            if is_test:
                print("\n[Science_Problem_Solver][Step 4 Final Output]")
                print(self._extract_content(final_outputs))
            return final_outputs
        except Exception as e:
            print(f"生成响应时出错(Science_Problem_Solver): {e}")
            return ""


