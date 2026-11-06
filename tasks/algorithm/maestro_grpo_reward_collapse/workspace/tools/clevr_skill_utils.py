from tools.erqa_solver_utils import (
    attach_usage_totals,
    accumulate_usage,
    build_messages,
    clean_output_content,
    collect_focus_crops,
    init_usage_totals,
    merge_usage_totals,
)


class BaseCLEVRSkillTool:
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

    def _build_prompt_text(self):
        return (
            f"You are solving a CLEVR {self.task_intro} question.\n\n"
            f"Question:\n{self.question}\n\n"
            "Requirements:\n"
            + "\n".join(f"- {item}" for item in self.requirements)
            + "\n"
        )

    def _build_per_image_extras(self, crop_bundle):
        crop_urls = crop_bundle.get("crop_urls", [])
        per_image_extras = []

        for idx in range(len(self.images)):
            extras = []
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
