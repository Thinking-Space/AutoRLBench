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


STEP2_PROMPT = """Think carefully and inspect the image step by step before answering.
List all the objects the question asked and give their exact positions.
Note that some objects may be occluded; do not overlook partially obscured or truncated objects.

Requirements:
- Focus only on the object category or categories requested in the question.
- Use the original image as the primary evidence.
- Use the detection image with bounding boxes and the detection result as helpful references, not as ground truth.
- Use the zoomed-in crop if provided to inspect crowded, tiny, or ambiguous regions.
- Deliberately scan the image region by region instead of answering immediately.
- Re-check crowded areas, borders, reflections, and partially visible objects before finalizing the list.
- Describe each matched object with an exact relative position such as left/right/top/bottom/center, and use finer-grained phrases when needed.
- If multiple similar objects appear, enumerate them clearly.
- If some objects are partially occluded or likely missed by detection, explicitly mention those suspected missed objects and their approximate positions.
"""


# STEP3_PROMPT = """Think carefully and verify the count before answering.
# Answer the question based on the object positions and the images.
# The position information may be incomplete; you need to identify objects that might have been missed.

# Requirements:
# - Use the original image as the primary evidence.
# - Use the detection image, detection result, and zoomed-in crop as supporting evidence.
# - Use the object-position list from the previous step, but do not trust it blindly.
# - Pause to check whether any object was double-counted or missed.
# - Reconcile conflicts between the previous list and the image evidence before giving the final answer.
# - If the previous step missed an object, correct the count based on the image.
# - Give concise reasoning.
# """

STEP3_PROMPT = """Think carefully and verify the count before answering.
Answer the question based on the object positions and the images.

Requirements:
- Use the original image as the primary evidence.
- Use the detection image, detection result, and zoomed-in crop as supporting evidence.
- Use the object-position list from the previous step, but do not trust it blindly.
- Pause to check whether any object was double-counted or missed.
- Reconcile conflicts between the previous list and the image evidence before giving the final answer.
- Give concise reasoning.
"""


class Counting_Problem_Solver_tool:
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

        self.detection_results_by_image = self._get_detection_results_list()
        self.detection_results_text = "\n\n".join(self.detection_results_by_image)
        self.box_images = self._get_box_images_for_paths()
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
            "1. Read detection result and boxed image.\n"
            "2. Use DeepEyes-7B to locate a region that needs closer inspection.\n"
            "3. Generate a zoomed-in crop for that region when needed.\n"
            "4. List all asked objects and their positions using the original image, boxed image, crop, and detection result.\n"
            "5. Answer the counting question based on the positions and image evidence.\n"
        )
        return [{"role": "user", "content": preview_text}]

    def _get_detection_results_file(self):
        return (
            "results/tool_results/detection.json"
            if self.dataset == ""
            else "results/tool_results/detection_eval.json"
        )

    def _load_detection_result_data(self):
        result_file = self._get_detection_results_file()
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取detection结果文件 {result_file} 时出错: {e}")
            return {}

    def _get_detection_results_list(self):
        if not self.image_paths:
            return []

        detection_data = self._load_detection_result_data()
        detection_results = []

        for i, image_path in enumerate(self.image_paths):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_key = image_name if self.dataset == "" else f"{self.dataset}_{image_name}"

            if image_key in detection_data and detection_data[image_key]:
                detection_results.append(
                    f"The detection result of image{i+1} is: {detection_data[image_key]}"
                )
            else:
                detection_results.append(f"The detection result of image{i+1} is: Not found")

        return detection_results

    def _get_box_image_path(self, image_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if self.dataset == "":
            base_path = f"data/images_with_box/{image_name}"
        else:
            base_path = f"data/{self.dataset}/images_with_boxes/{image_name}"

        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            candidate = base_path + ext
            if os.path.exists(candidate):
                return candidate
        return ""

    def _encode_file_to_data_url(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")

        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _get_box_images_for_paths(self):
        box_images = []
        for image_path in self.image_paths:
            box_image_path = self._get_box_image_path(image_path)
            box_images.append(self._encode_file_to_data_url(box_image_path) if box_image_path else "")
        return box_images

    def _build_messages(self, prompt_text, crop_images=None, include_images=True):
        if self.system_prompt == "":
            messages = []
        else:
            messages = [{"role": "system", "content": self.system_prompt}]

        if include_images and self.images:
            content = []
            crop_images = crop_images or []
            for idx, image in enumerate(self.images):
                content.append({"type": "image_url", "image_url": {"url": f"{image}"}})

                if idx < len(self.box_images) and self.box_images[idx]:
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"{self.box_images[idx]}"}}
                    )

                if idx < len(crop_images) and crop_images[idx]:
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"{crop_images[idx]}"}}
                    )

            content.append({"type": "text", "text": prompt_text})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt_text})

        return messages

    def _call_model(self, prompt_text, crop_images=None, include_images=True):
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt_text, crop_images=crop_images, include_images=include_images),
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

    def _get_deepeyes_decision(self, img_url, box_img_url="", detection_text=""):
        if self.deepeyes_client is None:
            print("DeepEyes-7B client is unavailable")
            return "", None

        user_prompt = (
            f"Question: {self.question}\n\n"
            f"Detection result hint:\n{detection_text or 'Not found'}\n\n"
            "The provided images are:\n"
            "1. The original image.\n"
            "2. The same image with detection boxes, if available.\n\n"
            "Think carefully about where counting may fail before responding. "
            "Identify the single local region that most needs magnification for accurate counting. "
            "Call **image_zoom_in_tool** only if zooming would help inspect a crowded, tiny, ambiguous, or partially occluded region. "
            "Do not rush to answer; first consider whether closer inspection is needed to avoid missing or double-counting objects. "
            "Then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>"
        )

        user_content = [{"type": "image_url", "image_url": {"url": img_url}}]
        if box_img_url:
            user_content.append({"type": "image_url", "image_url": {"url": box_img_url}})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": DEEPEYES_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
            print(f"DeepEyes-7B 调用失败(Counting): {e}")
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
                    saved_path = os.path.join(save_dir, f"{image_name}_{timestamp}_counting_crop.jpg")
                    crop.save(saved_path, format="JPEG")

                buffered = io.BytesIO()
                crop.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_str}", saved_path
        except Exception as e:
            print(f"裁剪图像失败: {e}")
            return None, None

    def _build_step2_prompt(self, has_crop):
        detection_text = self.detection_results_text if self.detection_results_text else "No detection result found."
        crop_text = (
            "A zoomed-in crop is also provided for images that needed closer inspection."
            if has_crop
            else "No zoomed-in crop was provided because no extra magnification was judged necessary."
        )
        return (
            "You are analyzing a counting question from one or more images.\n\n"
            "Think step by step and inspect the visual evidence carefully before writing the object list.\n\n"
            f"Question:\n{self.question}\n\n"
            "Detection result:\n"
            f"{detection_text}\n\n"
            f"{crop_text}\n\n"
            "For each image, the visual inputs are ordered as: original image, detection-box image, then zoomed crop if available.\n\n"
            "Step 2 instruction:\n"
            f"{STEP2_PROMPT}\n"
        )

    def _build_step3_prompt(self, step2_response, has_crop):
        detection_text = self.detection_results_text if self.detection_results_text else "No detection result found."
        crop_text = (
            "A zoomed-in crop is also provided for images that needed closer inspection."
            if has_crop
            else "No zoomed-in crop was provided because no extra magnification was judged necessary."
        )
        return (
            "You are solving a counting question from one or more images.\n\n"
            "Think step by step and verify the count carefully before giving the final answer.\n\n"
            f"Question:\n{self.question}\n\n"
            "Detection result:\n"
            f"{detection_text}\n\n"
            f"{crop_text}\n\n"
            "Object position list from Step 2:\n"
            f"{step2_response}\n\n"
            "For each image, the visual inputs are ordered as: original image, detection-box image, then zoomed crop if available.\n\n"
            "Step 3 instruction:\n"
            f"{STEP3_PROMPT}\n"
        )

    def generate(self, is_test=False):
        try:
            usage_totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            if self.client is None:
                print("Counting_Problem_Solver: client is None")
                return ""
            if not self.images:
                print("Counting_Problem_Solver: no images provided")
                return ""

            crop_images = ["" for _ in self.images]

            if is_test:
                print("\n[Counting_Problem_Solver][Step 1 Detection Result]")
                print(self.detection_results_text)
                print("\n[Counting_Problem_Solver][Box Images Loaded]")
                print(len([img for img in self.box_images if img]))

            for idx, image_url in enumerate(self.images):
                image_path = self.image_paths[idx] if idx < len(self.image_paths) else None
                box_image_url = self.box_images[idx] if idx < len(self.box_images) else ""
                detection_text = (
                    self.detection_results_by_image[idx]
                    if idx < len(self.detection_results_by_image)
                    else f"The detection result of image{idx+1} is: Not found"
                )

                deepeyes_res, deepeyes_outputs = self._get_deepeyes_decision(
                    image_url,
                    box_img_url=box_image_url,
                    detection_text=detection_text,
                )
                self._accumulate_usage(usage_totals, deepeyes_outputs)

                if is_test:
                    print(f"\n[Counting_Problem_Solver][Step 2 DeepEyes Output][image{idx+1}]")
                    print(deepeyes_res)

                if "<tool_call>" in deepeyes_res and image_path:
                    save_dir = "tools/test_zoom" if is_test else None
                    crop_url, saved_path = self._crop_image(image_path, deepeyes_res, save_dir=save_dir)
                    if crop_url:
                        crop_images[idx] = crop_url
                    if is_test and saved_path:
                        print(f"[Counting_Problem_Solver] Cropped image saved to: {saved_path}")

            has_crop = any(crop_images)

            step2_outputs = self._call_model(
                self._build_step2_prompt(has_crop=has_crop),
                crop_images=crop_images,
                include_images=True,
            )
            self._accumulate_usage(usage_totals, step2_outputs)
            step2_response = self._remove_think_content(self._extract_content(step2_outputs))
            if is_test:
                print("\n[Counting_Problem_Solver][Step 3 Object Listing Output]")
                print(step2_response)

            step3_outputs = self._call_model(
                self._build_step3_prompt(step2_response, has_crop=has_crop),
                crop_images=crop_images,
                include_images=True,
            )
            self._accumulate_usage(usage_totals, step3_outputs)
            step3_outputs = self._clean_output_content(step3_outputs)
            step3_outputs = self._attach_usage_totals(step3_outputs, usage_totals)
            if is_test:
                print("\n[Counting_Problem_Solver][Step 4 Final Output]")
                print(self._extract_content(step3_outputs))
            return step3_outputs
        except Exception as e:
            print(f"生成响应时出错(Counting_Problem_Solver): {e}")
            return ""


