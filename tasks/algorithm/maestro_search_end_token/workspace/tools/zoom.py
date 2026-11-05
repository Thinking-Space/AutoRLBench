import json
import os
import re
import io
import base64
import time
from PIL import Image
from tools.api_client import OpenAI
from tools.service_config import get_service

class Zoom_tool:
    def __init__(self, tool, client, model, system_prompt, question, images, temperature, top_p, timeout, have_image=False, image_paths=[], dataset=""):
        self.tool = tool
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.question = question
        self.images = images  # Base64 字符串列表
        self.have_image = have_image
        self.image_paths = image_paths
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.dataset = dataset
        
        service = get_service('deepeyes')
        self.deepeyes_base_url = service.base_url
        self.deepeyes_model_id = service.served_model_name
        self.deepeyes_client = OpenAI(base_url=self.deepeyes_base_url, api_key="EMPTY")

    def _get_deepeyes_decision(self, img_url):
        """阶段 1: 调用 DeepEyes-7B 判断是否需要 Zoom"""
        instruction_prompt_system = """You are a helpful assistant.
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}},"required":["bbox_2d"]}}}
</tools>
# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}
</tool_call>"""

        user_prompt_v2 = f"\nQuestion: {self.question}\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) <answer>...</answer>"

        messages = [
            {"role": "system", "content": instruction_prompt_system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": user_prompt_v2}
            ]}
        ]

        try:
            response = self.deepeyes_client.chat.completions.create(
                model=self.deepeyes_model_id,
                messages=messages,
                temperature=0.0,
                stop=["<|im_end|>\n", "</answer>", "</tool_call>"],
                timeout=self.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepEyes-7B 调用失败: {e}")
            return ""

    def _crop_image(self, image_path, deepeyes_output, save_dir=None):
        """执行物理裁剪"""
        try:
            call_str = deepeyes_output.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            try:
                action = json.loads(call_str.replace("'", '"'))
            except:
                action = eval(call_str)
            
            bbox = action['arguments']['bbox_2d']
            
            with Image.open(image_path) as img:
                crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                saved_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    timestamp = int(time.time() * 1000)
                    saved_path = os.path.join(save_dir, f"{image_name}_{timestamp}_crop.jpg")
                    crop.save(saved_path, format="JPEG")
                buffered = io.BytesIO()
                crop.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{img_str}", saved_path
        except Exception as e:
            print(f"裁剪图像失败: {e}")
            return None, None

    def generate(self, is_test=False):
        """核心生成逻辑"""
        if not self.images: return None
            
        main_img_url = self.images[0]
        main_img_path = self.image_paths[0] if self.image_paths else None

        # 1. 询问预审模型
        # print(f"--- [DeepEyes-7B] 正在预审图像... ---")
        # import pdb; pdb.set_trace()
        # print(1)
        deepeyes_res = self._get_deepeyes_decision(main_img_url)
        # print(2)
        # import pdb; pdb.set_trace()
        if is_test:
            print("\n[Zoom Test] DeepEyes intermediate output:")
            print(deepeyes_res)
        
        final_content = []
        
        # 2. 判断是否缩放
        if '<tool_call>' in deepeyes_res and main_img_path:
            # print(f"--- [DeepEyes-7B] 发现关键区域，执行裁剪... ---")
            save_dir = "tools/test_zoom" if is_test else None
            crop_url, saved_path = self._crop_image(main_img_path, deepeyes_res, save_dir=save_dir)
            if is_test and saved_path:
                print(f"[Zoom Test] Cropped image saved to: {saved_path}")
            
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
            # print(f"--- [DeepEyes-7B] 无需放大，直接询问主模型 ---")
            final_content.append({"type": "image_url", "image_url": {"url": main_img_url}})
            final_content.append({"type": "text", "text": self.question})

        # 3. 请求主模型 GLM-4.6V-Flash
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        else:
            system_content = "You are an assistant who is good at observation and skilled at judging colors and the spatial positions of objects. When you need to determine the color of an object or the relative position between objects, carefully analyze the image, identify which part corresponds to the target object, and then answer the question."
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": final_content})

        try:
            # import pdb; pdb.set_trace()
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self.timeout
            )
            # print(3)
            # import pdb; pdb.set_trace()
            if is_test:
                print("\n[Zoom Test] Main model output:")
                if hasattr(outputs, 'choices') and outputs.choices:
                    print(outputs.choices[0].message.content)
                else:
                    print(outputs)
            return outputs
        except Exception as e:
            return f"GLM模型响应出错: {e}"

# ================= 测试入口 =================
