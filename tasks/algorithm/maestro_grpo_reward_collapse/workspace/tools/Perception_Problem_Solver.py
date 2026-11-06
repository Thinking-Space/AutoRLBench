import base64
import io
import json
import os
import time
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


class Perception_Problem_Solver_tool:
    def __init__(self, tool, client, model, system_prompt, question, images, temperature, top_p, timeout, have_image=False, image_paths=[], dataset=""):
        self.tool = tool
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.question = question
        self.images = images
        self.have_image = have_image
        self.image_paths = image_paths if image_paths else []
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.dataset = dataset
        self.messages = [{"role": "user", "content": self.question}]

        service = get_service('deepeyes')
        self.deepeyes_base_url = service.base_url
        self.deepeyes_model_id = service.served_model_name
        self.deepeyes_client = None
        if OpenAI is not None:
            try:
                self.deepeyes_client = OpenAI(base_url=self.deepeyes_base_url, api_key="EMPTY")
            except Exception as e:
                print(f"初始化 DeepEyes-7B 客户端失败: {e}")

    def _get_deepeyes_decision(self, img_url):
        instruction_prompt_system = """You are a helpful assistant.
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}},"required":["bbox_2d"]}}}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}
</tool_call>"""

        user_prompt = (
            f"\nQuestion: {self.question}\n"
            "Think first, call **image_zoom_in_tool** if needed, then answer. "
            "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>"
        )

        messages = [
            {"role": "system", "content": instruction_prompt_system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": user_prompt}
            ]}
        ]

        if self.deepeyes_client is None:
            print("DeepEyes-7B client is unavailable")
            return "", None

        try:
            # import pdb; pdb.set_trace()
            response = self.deepeyes_client.chat.completions.create(
                model=self.deepeyes_model_id,
                messages=messages,
                temperature=0.0,
                stop=["<|im_end|>\n", "</answer>", "</tool_call>"],
                timeout=self.timeout
            )
            return response.choices[0].message.content, response
        except Exception as e:
            print(f"DeepEyes-7B 调用失败(Perception): {e}")
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
                action = eval(call_str)

            bbox = action["arguments"]["bbox_2d"]

            with Image.open(image_path) as img:
                crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                saved_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    timestamp = int(time.time() * 1000)
                    saved_path = os.path.join(save_dir, f"{image_name}_{timestamp}_perception_crop.jpg")
                    crop.save(saved_path, format="JPEG")
                buffered = io.BytesIO()
                crop.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_str}", saved_path
        except Exception as e:
            print(f"裁剪图像失败: {e}")
            return None, None

    def _remove_think_content(self, text):
        if not isinstance(text, str) or text == "":
            return text
        import re

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

    def generate(self, is_test=False):
        if not self.images:
            return None

        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        main_img_url = self.images[0]
        main_img_path = self.image_paths[0] if self.image_paths else None

        deepeyes_res, deepeyes_outputs = self._get_deepeyes_decision(main_img_url)
        self._accumulate_usage(usage_totals, deepeyes_outputs)
        if is_test:
            print("\n[Perception_Problem_Solver][DeepEyes Output]")
            print(deepeyes_res)

        final_content = []

        if "<tool_call>" in deepeyes_res and main_img_path:
            save_dir = "tools/test_zoom" if is_test else None
            crop_url, saved_path = self._crop_image(main_img_path, deepeyes_res, save_dir=save_dir)
            if is_test and saved_path:
                print(f"[Perception_Problem_Solver] Cropped image saved to: {saved_path}")

            prompt_text = (
                f"Question: {self.question}\n\n"
                "To answer accurately, I've provided:\n"
                "1. Global View (Original)\n"
                "2. Focused View (Zoomed-in Crop)\n"
                "Analyze both to provide your final answer."
            )

            final_content.append({"type": "image_url", "image_url": {"url": main_img_url}})
            if crop_url:
                final_content.append({"type": "image_url", "image_url": {"url": crop_url}})
            final_content.append({"type": "text", "text": prompt_text})
        else:
            final_content.append({"type": "image_url", "image_url": {"url": main_img_url}})
            final_content.append({"type": "text", "text": self.question})

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        else:
            system_content = "You are an assistant who is good at observation and skilled at judging colors and the spatial positions of objects. When you need to determine the color of an object or the relative position between objects, carefully analyze the image, identify which part corresponds to the target object, and then answer the question."
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": final_content})

        try:
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self.timeout
            )
            self._accumulate_usage(usage_totals, outputs)
            # outputs = self._clean_output_content(outputs)
            outputs = self._attach_usage_totals(outputs, usage_totals)
            if is_test:
                print("\n[Perception_Problem_Solver][Main Model Output]")
                if hasattr(outputs, "choices") and outputs.choices:
                    print(outputs.choices[0].message.content)
                else:
                    print(outputs)
            return outputs
        except Exception as e:
            return f"GLM模型响应出错: {e}"


