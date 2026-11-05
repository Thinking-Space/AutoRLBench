import json
import glob
import os

class Yolo_tool:
    def __init__(self, tool, client, model, system_prompt, question, images, temperature, top_p, timeout, have_image=False, image_paths=[]):
        self.tool = tool
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image
        self.image_paths = image_paths if image_paths else []

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
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                })
            
            # 读取OCR结果并添加到消息中
            yolo_results_text = self._get_yolo_results_for_images()
            if yolo_results_text:
                # 添加OCR结果文本
                # content.append({"type": "text", "text": f"{yolo_results_text}"})
                pass
            
            # 添加问题文本
            prompt = f"Objects detection results: {yolo_results_text} \nAnswer the question based on the image. And take objects detection results as a reference. The objects detection results may be less but are impossible to be more. So your answer should more than the number that have been detected.\n Question:\n"
            content.append({"type": "text", "text": prompt + question})
            
            messages.append({"role": "user", "content": content})
        # import pdb; pdb.set_trace()
        self.messages = messages

    def _get_yolo_results_for_images(self):
        """根据当前图片路径获取对应的OCR结果"""
        if not self.image_paths:
            return ""
        
        # 读取results/yolo_results路径下所有的json文件
        yolo_results_dir = "results/yolo_results"
        yolo_data = {}
        
        # 查找所有json文件
        json_files = glob.glob(os.path.join(yolo_results_dir, "*.json"))
        
        # 从所有json文件中读取OCR结果
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 假设每个json文件是一个列表，包含多个OCR结果
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "image_path" in item and "yolo_result" in item:
                                # 使用绝对路径作为key
                                abs_path = os.path.abspath(item["image_path"])
                                yolo_data[abs_path] = item["yolo_result"]
            except Exception as e:
                print(f"读取OCR结果文件 {json_file} 时出错: {e}")
                continue
        
        # 构建当前图片的OCR结果文本
        yolo_results = []
        for i, image_path in enumerate(self.image_paths):
            abs_path = os.path.abspath(image_path)
            if abs_path in yolo_data and yolo_data[abs_path]:
                # 只添加成功的OCR结果
                yolo_results.append(f"The size of image{i+1} and objects with positions in image{i+1} are: {yolo_data[abs_path]}")
            else:
                yolo_results.append(f"The size of image{i+1} and objects with positions in image{i+1} are: Not found")
        
        return "\n\n".join(yolo_results) if yolo_results else ""

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
            print(f"生成响应时出错: {e}")
            return ""