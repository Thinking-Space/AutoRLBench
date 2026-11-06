import json

from tools.Perception_Problem_Solver import Perception_Problem_Solver_tool


class Perception_Problem_Solver_Eval_tool(Perception_Problem_Solver_tool):
    CATEGORY_GENERAL = "general"
    CATEGORY_OBJECT_COLOR = "object_color"
    CATEGORY_RELATIVE_POSITION = "relative_position"
    CATEGORY_MEDICAL = "medical"
    CATEGORY_COUNTING = "counting"
    CATEGORY_TABLE = "table"

    SUPPORTED_CATEGORIES = (
        CATEGORY_OBJECT_COLOR,
        CATEGORY_RELATIVE_POSITION,
        CATEGORY_MEDICAL,
        CATEGORY_COUNTING,
        CATEGORY_TABLE,
    )

    CATEGORY_SYSTEM_PROMPTS = {
        CATEGORY_GENERAL: (
            "You are a careful visual question answering assistant. "
            "Use the image and the question together, identify the relevant objects or regions, "
            "reason from visible evidence only, and answer concisely and accurately. "
            "If the question is multiple choice, choose the option best supported by the image."
        ),
        CATEGORY_OBJECT_COLOR: (
            "You are an assistant who is good at observation and skilled at judging colors of objects. When you need to determine the color of an object or the relative position between objects, carefully analyze the image, identify which part corresponds to the target object, and then answer the question."
        ),
        CATEGORY_RELATIVE_POSITION: (
            "You are an assistant who is good at observation and skilled at judging the spatial positions of objects. When you need to determine the color of an object or the relative position between objects, carefully analyze the image, identify which part corresponds to the target object, and then answer the question."
        ),
        CATEGORY_MEDICAL: (
            "You are an assistant who is careful with medical visual questions. "
            "Examine the image step by step, focusing on medically relevant structures, abnormalities, stains, morphology, and context. "
            "Base the answer on visible evidence only, avoid unsupported claims, and choose the best diagnosis or description supported by the image. "
            "If the question is multiple choice, select the option that best matches the medical evidence. "
            "Reply concisely and accurately."
        ),
        CATEGORY_COUNTING: (
            "You are an assistant who is expert at counting objects in images. "
            "Identify the target object category first, then count methodically from one side of the image to the other. "
            "Avoid double-counting repeated, overlapping, reflected, or partially occluded objects, and include only instances that truly belong to the asked category. "
            "If needed, recount before answering. "
            "Reply with the final count only when the question asks for a number."
        ),
        CATEGORY_TABLE: (
            "You are an assistant who is expert at answering table-based visual questions. "
            "Read the table structure carefully, including titles, row labels, column headers, units, and the target cell intersection. "
            "Do not confuse similar row names, column names, or nearby cells. "
            "If the answer requires comparison, compare only the relevant entries from the table. "
            "Reply concisely and faithfully to the table contents."
        ),
    }

    def _extract_text_from_response(self, response):
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, "choices") and response.choices:
            content = getattr(response.choices[0].message, "content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                return "\n".join(part for part in text_parts if part).strip()
        return str(response).strip()

    def _classify_question(self):
        if not self.images:
            return self.CATEGORY_GENERAL
        return self._classify_question_with_usage(self.images[0])[0]

    def _classify_question_with_usage(self, img_url):
        if self.client is None:
            return self.CATEGORY_GENERAL, None

        classify_system_prompt = (
            "You are a classifier for visual question answering tasks. "
            "You must classify the task using both the image and the question. "
            "Classify the question into exactly one of these categories: "
            "object_color, relative_position, medical, counting, table. "
            "Return JSON only, in the format {\"category\": \"<one category>\"}. "
            "Use object_color for questions asking about the color of a specific object, "
            "relative_position for spatial relation questions, "
            "medical for medical or pathology questions, counting for counting questions, "
            "and table for questions about reading a table. "
            "Do not use keyword matching shortcuts; infer the category from the visual content and the question together."
        )
        classify_messages = [
            {"role": "system", "content": classify_system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": f"Question: {self.question}"},
            ]},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=classify_messages,
                temperature=0.0,
                top_p=1.0,
                timeout=self.timeout,
            )
            text = self._extract_text_from_response(response)
            try:
                parsed = json.loads(text)
                category = parsed.get("category", "")
                if category in self.SUPPORTED_CATEGORIES:
                    return category, response
            except Exception:
                pass

            for category in self.SUPPORTED_CATEGORIES:
                if category in text:
                    return category, response
        except Exception as e:
            print(f"Question classification failed, falling back to general category: {e}")

        return self.CATEGORY_GENERAL, response if "response" in locals() else None

    def _build_system_prompt(self, category):
        category_prompt = self.CATEGORY_SYSTEM_PROMPTS.get(category, self.CATEGORY_SYSTEM_PROMPTS[self.CATEGORY_GENERAL])
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{category_prompt}"
        return category_prompt

    def _build_final_prompt_text(self, category, include_crop):
        if include_crop:
            prompt_lines = [
                f"Question: {self.question}",
                "",
                f"Question category: {category}",
                "To answer accurately, I've provided:",
                "1. Global View (Original)",
                "2. Focused View (Zoomed-in Crop)",
            ]

            # if category == self.CATEGORY_OBJECT_COLOR:
            #     prompt_lines.extend([
            #         "For color questions, first check whether the zoomed-in crop actually contains the target object mentioned in the question.",
            #         "If it does, use that crop to judge the color more precisely.",
            #         "If it does not clearly contain the target object, use the original image to find the correct object and do not answer based on the crop alone.",
            #     ])

            prompt_lines.append("Analyze the provided image views step by step and give the final answer.")
            return "\n".join(prompt_lines)

        return f"Question category: {category}\nQuestion: {self.question}"

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

        question_category, classify_outputs = self._classify_question_with_usage(main_img_url)
        # print(question_category)
        self._accumulate_usage(usage_totals, classify_outputs)
        if is_test:
            print("\n[Perception_Problem_Solver_Eval][Question Category]")
            print(question_category)

        deepeyes_res, deepeyes_outputs = self._get_deepeyes_decision(main_img_url)
        self._accumulate_usage(usage_totals, deepeyes_outputs)
        if is_test:
            print("\n[Perception_Problem_Solver_Eval][DeepEyes Output]")
            print(deepeyes_res)

        final_content = []

        if "<tool_call>" in deepeyes_res and main_img_path:
            save_dir = "tools/test_zoom" if is_test else None
            crop_url, saved_path = self._crop_image(main_img_path, deepeyes_res, save_dir=save_dir)
            if is_test and saved_path:
                print(f"[Perception_Problem_Solver_Eval] Cropped image saved to: {saved_path}")

            prompt_text = self._build_final_prompt_text(question_category, include_crop=bool(crop_url))

            final_content.append({"type": "image_url", "image_url": {"url": main_img_url}})
            if crop_url:
                final_content.append({"type": "image_url", "image_url": {"url": crop_url}})
            final_content.append({"type": "text", "text": prompt_text})
        else:
            final_content.append({"type": "image_url", "image_url": {"url": main_img_url}})
            final_content.append({"type": "text", "text": self._build_final_prompt_text(question_category, include_crop=False)})

        messages = [
            {"role": "system", "content": self._build_system_prompt(question_category)},
            {"role": "user", "content": final_content},
        ]

        try:
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self.timeout,
            )
            self._accumulate_usage(usage_totals, outputs)
            outputs = self._clean_output_content(outputs)
            outputs = self._attach_usage_totals(outputs, usage_totals)
            if is_test:
                print("\n[Perception_Problem_Solver_Eval][Main Model Output]")
                if hasattr(outputs, "choices") and outputs.choices:
                    print(outputs.choices[0].message.content)
                else:
                    print(outputs)
            return outputs
        except Exception as e:
            return f"GLM模型响应出错: {e}"


