from tools.deepscaler_rm import extract_answer
from collections import Counter


class Self_consistency_tool:
    def __init__(self, tool, client, model, system_prompt, question, images, temperature, top_p, timeout, have_image=False):
        self.tool = tool
        self.client = client
        self.model = model
        # self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image
        self.number = tool.tool_config['number']

        if images == []:
            messages=[{"role": "user", "content": question}]
        else:
            messages_text=[{"role": "user", "content": question}]
            # 构建包含图片的消息内容
            if system_prompt == "":
                messages = []
            else:
                messages = [{"role": "system", "content": system_prompt}]
            content = []

            # content = [{"type": "text", "text": "Describe this picture."}]
            
            # 添加所有图片
            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    }
                })
            content.append({"type": "text", "text": question})
            
            messages.append({"role": "user", "content": content})
        self.messages = messages

    def generate(self):
        try:
            # 获取多个响应
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self.timeout,
                n=self.number,
            )
            
            # 提取每个响应的答案
            answers = []
            for output in outputs['choices']:
                answer = extract_answer(output['message']['content'])  # 提取答案
                answers.append((answer, output))  # 存储答案和响应对象

            # 计算一致性，返回频率最高的答案对应的完整响应
            consistency_output = self.calculate_consistency(answers)
            
            return consistency_output
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return ""

    def calculate_consistency(self, answers):
        # 统计每个答案的频率
        counter = Counter([answer[0] for answer in answers])
        
        # 获取出现频率最高的答案及其频率
        most_common_answer, frequency = counter.most_common(1)[0]
        
        # 返回与最常见答案对应的完整响应对象
        for answer, output in answers:
            if answer == most_common_answer:
                return output  # 返回整个响应
