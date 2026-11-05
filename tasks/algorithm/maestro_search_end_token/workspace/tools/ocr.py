import json
import os

class Ocr_tool:
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

            # 添加所有图片
            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image}"
                    }
                })
            
            # 读取OCR结果并添加到消息中
            ocr_results_text = self._get_ocr_results_for_images()
            # print(f"[DEBUG] ocr_results: {ocr_results_text}")
            if ocr_results_text:
                # 添加OCR结果文本
                content.append({"type": "text", "text": f"OCR results:\n{ocr_results_text[:3000]}"})
            
            # 添加问题文本
            prompt = "Answer the question based on the ocr result and the image.\n Question:\n"
            content.append({"type": "text", "text": prompt + question})
            
            messages.append({"role": "user", "content": content})
        # import pdb; pdb.set_trace()
        self.messages = messages

    def _get_ocr_results_for_images(self):
        """根据当前图片路径获取对应的OCR结果"""
        if not self.image_paths:
            return ""

        # 读取OCR结果文件
        if self.dataset == "":
            ocr_results_file = "results/tool_results/ocr.json"
        else:
            ocr_results_file = "results/tool_results/ocr_eval.json"
        ocr_data = {}

        # 读取单个JSON文件
        try:
            with open(ocr_results_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
        except Exception as e:
            print(f"读取OCR结果文件 {ocr_results_file} 时出错: {e}")
            return ""

        # 构建当前图片的OCR结果文本
        ocr_results = []
        for i, image_path in enumerate(self.image_paths):
            # 提取图片名称（不包括后缀），这个名称是一个数字
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            if self.dataset == "":
                image_key = image_name  # 直接使用图片名称作为key
            else:
                image_key = f"{self.dataset}_{image_name}"

            if image_key in ocr_data and ocr_data[image_key]:
                # 只添加成功的OCR结果
                ocr_results.append(f"The ocr result of image{i+1} is: {ocr_data[image_key]}")
            else:
                ocr_results.append(f"The ocr result of image{i+1} is: Not found")

        return "\n\n".join(ocr_results) if ocr_results else ""

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
            print(f"生成响应时出错(ocr): {e}")
            return ""