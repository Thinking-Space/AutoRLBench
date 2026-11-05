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


DEFAULT_SYSTEM_PROMPT = (
    "You solve embodied manipulation questions. "
    "Use the image as primary evidence. "
    "Infer the smallest corrective action that best aligns the gripper, object, or camera with the target."
)


class ERQA_Action_Adjustment_Solver_tool:
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

        self.caption_results_text = format_result_for_images(
            "caption", self.image_paths, dataset=self.dataset, caption_empty_default=True
        )
        self.detection_results_text = format_result_for_images(
            "detection", self.image_paths, dataset=self.dataset
        )
        self.box_images = get_box_images_for_paths(self.image_paths, dataset=self.dataset)

    def generate(self, is_test=False):
        if not self.images:
            return None

        usage_totals = init_usage_totals()
        crop_bundle = collect_focus_crops(
            question=self.question,
            images=self.images,
            image_paths=self.image_paths,
            timeout=self.timeout,
            focus_hint="Focus on the gripper, fingertips, handle, slot, opening, object edge, and the local target alignment.",
            max_images=1,
            crop_suffix="action_crop",
            is_test=is_test,
        )
        merge_usage_totals(usage_totals, crop_bundle["usage_totals"])

        per_image_extras = []
        for idx in range(len(self.images)):
            extras = []
            if idx < len(self.box_images) and self.box_images[idx]:
                extras.append(self.box_images[idx])
            if idx < len(crop_bundle["crop_urls"]) and crop_bundle["crop_urls"][idx]:
                extras.append(crop_bundle["crop_urls"][idx])
            per_image_extras.append(extras)

        caption_text = self.caption_results_text or "No caption result found."
        detection_text = self.detection_results_text or "No detection result found."
        prompt_text = (
            "You are solving an ERQA action adjustment question.\n\n"
            f"Question:\n{self.question}\n\n"
            "Caption evidence:\n"
            f"{caption_text}\n\n"
            # "Detection evidence:\n"
            # f"{detection_text}\n\n"
            "Requirements:\n"
            "- Determine the correct image coordinate frame.\n"
            "- Compare the current pose with the target and identify the main misalignment.\n"
            "- Choose the smallest corrective motion or rotation that would make the action feasible.\n"
            "- The final line must be only the answer in the exact format requested by the question.\n"
        )
        messages = build_messages(
            system_prompt=self.system_prompt or DEFAULT_SYSTEM_PROMPT,
            images=self.images,
            prompt_text=prompt_text,
            # per_image_extras=per_image_extras,
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
