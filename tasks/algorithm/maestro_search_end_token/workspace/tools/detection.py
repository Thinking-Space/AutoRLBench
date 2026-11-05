import json
import os
import base64
from io import BytesIO
from tools.api_client import OpenAI
from PIL import Image
from tools.erqa_solver_utils import get_box_image_path

class Detection_tool:
    def __init__(self, tool, client, model, system_prompt, question, images, temperature, top_p, timeout, have_image=False, image_paths=[], dataset=""):
        self.tool = tool
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image
        self.image_paths = image_paths if image_paths else []
        self.dataset = dataset

        if images == []:
            messages = [{"role": "user", "content": question}]
        else:
            # 构建包含图片的消息内容
            if system_prompt == "":
                messages = []
            else:
                messages = [{"role": "system", "content": system_prompt}]
            content = []

            # 添加所有图片（原始图片和检测框图片）
            for i, image in enumerate(images):
                # 添加原始图片
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image}"
                    }
                })

                # 添加对应的检测框图片
                if i < len(self.image_paths):
                    image_path = self.image_paths[i]
                    # 提取图片名称（不包括后缀）
                    box_image_path = get_box_image_path(image_path, dataset=self.dataset)
                    box_image_base64 = None
                    if box_image_path:
                        box_image_base64 = self._image_to_base64(box_image_path, max_side=1024)
                    else:
                        print(f"Warning: No boxed image for {image_path}; using the original image and detection text")
                    if box_image_base64:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": box_image_base64
                            }
                        })

            # 读取检测结果并添加到消息中
            detection_results_text = self._get_detection_results_for_images()
            # print(f"[DEBUG] detection_results_text: {detection_results_text}")
            if detection_results_text:
                # 添加检测结果文本
                content.append({"type": "text", "text": f"Detection results:\n{detection_results_text[:3000]}"})

            # 添加问题文本
            prompt = "Answer the question based on the detection result and the images. Where available, each image pair consists of: 1) the original image, and 2) the same image with detection bounding boxes drawn. Note that the detection results may be incomplete; some objects might not have been detected. \n Question:\n"
            content.append({"type": "text", "text": prompt + question})

            messages.append({"role": "user", "content": content})
        # import pdb; pdb.set_trace()
        self.messages = messages

    def _image_to_base64(self, image_path, max_side=None):
        """将图片文件转换为base64格式"""
        # try:
        if 1:
            with Image.open(image_path) as img:
                width, height = img.size
                longest_side = max(width, height)
                if max_side is not None and longest_side > max_side:
                    scale = max_side / longest_side
                    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # 统一转换为 RGB，并以 JPEG 编码为 data URL。
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                buffer = BytesIO()
                img.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
                base64_string = base64.b64encode(image_bytes).decode('utf-8')

                return f"data:image/jpeg;base64,{base64_string}"
        # except Exception as e:
        #     print(f"读取图片文件 {image_path} 时出错: {e}")
        #     return None

    def _get_detection_results_for_images(self):
        """根据当前图片路径获取对应的检测结果"""
        if not self.image_paths:
            return ""

        # 读取检测结果文件
        if self.dataset == "":
            detection_results_file = "results/tool_results/detection.json"
        else:
            detection_results_file = "results/tool_results/detection_eval.json"
        detection_data = {}

        # 读取单个JSON文件
        try:
            with open(detection_results_file, 'r', encoding='utf-8') as f:
                detection_data = json.load(f)
        except Exception as e:
            print(f"读取检测结果文件 {detection_results_file} 时出错: {e}")
            return ""

        # 构建当前图片的检测结果文本
        detection_results = []
        for i, image_path in enumerate(self.image_paths):
            # 提取图片名称（不包括后缀），这个名称是一个数字
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            if self.dataset == "":
                image_key = image_name  # 直接使用图片名称作为key
            else:
                image_key = f"{self.dataset}_{image_name}"

            if image_key in detection_data and detection_data[image_key]:
                # 只添加成功的检测结果
                detection_results.append(f"The detection result of image{i+1} is: {detection_data[image_key]}")
            else:
                detection_results.append(f"The detection result of image{i+1} is: Not found")

        return "\n\n".join(detection_results) if detection_results else ""

    def generate(self):
        try:
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self.timeout
            )
            return outputs
        except Exception as e:
            print(f"生成响应时出错(detection) {self.model} \n {self.messages[-1]['content'][-1]}\n\n: {e}")
            return ""


# ================= 测试入口 =================
