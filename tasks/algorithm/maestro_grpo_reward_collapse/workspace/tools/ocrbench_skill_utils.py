from tools.erqa_solver_utils import (
    attach_usage_totals,
    accumulate_usage,
    build_messages,
    clean_output_content,
    collect_focus_crops,
    format_result_for_images,
    get_box_images_for_paths,
    init_usage_totals,
    merge_usage_totals,
)


class BaseOCRBenchSkillTool:
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
        image_paths=None,
        dataset="",
        *,
        skill_name,
        default_system_prompt,
        task_intro,
        requirements,
        focus_hint="",
        max_crop_images=1,
        use_ocr=True,
        use_caption=False,
        use_detection=False,
        include_box_images=False,
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

        self.skill_name = skill_name
        self.default_system_prompt = default_system_prompt
        self.task_intro = task_intro
        self.requirements = requirements
        self.focus_hint = focus_hint
        self.max_crop_images = max_crop_images
        self.use_ocr = use_ocr
        self.use_caption = use_caption
        self.use_detection = use_detection
        self.include_box_images = include_box_images

        self.ocr_results_text = format_result_for_images(
            "ocr", self.image_paths, dataset=self.dataset
        )
        self.caption_results_text = format_result_for_images(
            "caption", self.image_paths, dataset=self.dataset, caption_empty_default=True
        )
        self.detection_results_text = format_result_for_images(
            "detection", self.image_paths, dataset=self.dataset
        )
        self.box_images = get_box_images_for_paths(self.image_paths, dataset=self.dataset)

    def _build_prompt_text(self):
        prompt_parts = [
            f"You are solving an OCRBench {self.task_intro} question.\n\n"
            f"Question:\n{self.question}\n"
        ]

        if self.use_ocr:
            prompt_parts.append(
                "OCR evidence:\n"
                f"{self.ocr_results_text or 'No OCR result found.'}\n"
            )
        if self.use_caption:
            prompt_parts.append(
                "Caption evidence:\n"
                f"{self.caption_results_text or 'No caption result found.'}\n"
            )
        if self.use_detection:
            prompt_parts.append(
                "Detection evidence:\n"
                f"{self.detection_results_text or 'No detection result found.'}\n"
            )

        prompt_parts.append(
            "Requirements:\n"
            + "\n".join(f"- {item}" for item in self.requirements)
            + "\n"
        )
        return "\n".join(prompt_parts)

    def _build_per_image_extras(self, crop_bundle):
        per_image_extras = []
        crop_urls = crop_bundle.get("crop_urls", [])

        for idx in range(len(self.images)):
            extras = []
            if self.include_box_images and idx < len(self.box_images) and self.box_images[idx]:
                extras.append(self.box_images[idx])
            if idx < len(crop_urls) and crop_urls[idx]:
                extras.append(crop_urls[idx])
            per_image_extras.append(extras)
        return per_image_extras

    def generate(self, is_test=False):
        if not self.images:
            return None

        usage_totals = init_usage_totals()
        crop_bundle = {
            "crop_urls": [""] * len(self.images),
            "saved_paths": [],
            "decisions": [],
            "usage_totals": init_usage_totals(),
        }

        if self.focus_hint and self.image_paths:
            crop_bundle = collect_focus_crops(
                question=self.question,
                images=self.images,
                image_paths=self.image_paths,
                timeout=self.timeout,
                focus_hint=self.focus_hint,
                max_images=min(self.max_crop_images, len(self.images)),
                crop_suffix=f"{self.skill_name.lower()}_crop",
                is_test=is_test,
            )
            merge_usage_totals(usage_totals, crop_bundle["usage_totals"])

        if is_test and crop_bundle.get("decisions"):
            for idx, decision in enumerate(crop_bundle["decisions"], start=1):
                print(f"[{self.skill_name}] focus decision {idx}: {decision}")

        messages = build_messages(
            system_prompt=self.system_prompt or self.default_system_prompt,
            images=self.images,
            prompt_text=self._build_prompt_text(),
            per_image_extras=self._build_per_image_extras(crop_bundle),
        )

        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
        )
        accumulate_usage(usage_totals, outputs)
        outputs = clean_output_content(outputs)
        return attach_usage_totals(outputs, usage_totals)
