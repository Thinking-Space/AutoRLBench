from tools.ocrbench_skill_utils import BaseOCRBenchSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve OCRBench text recognition tasks. "
    "Read the visible text faithfully and output only the exact recognized string."
)


class OCRBench_Text_Recognition_Skill_tool(BaseOCRBenchSkillTool):
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
        super().__init__(
            tool=tool,
            client=client,
            model=model,
            system_prompt=system_prompt,
            question=question,
            images=images,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            have_image=have_image,
            image_paths=image_paths,
            dataset=dataset,
            skill_name="OCRBench_Text_Recognition_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="text recognition",
            requirements=[
                "Treat this as faithful transcription, not semantic guessing.",
                "Read the main text string or number carefully at character level.",
                "Preserve casing and digits exactly when they are visible.",
                "Do not explain the answer. The final line must be only the recognized text.",
            ],
            focus_hint="Focus tightly on the main text string, including ambiguous characters such as O/0, I/1, S/5, and B/8.",
            max_crop_images=1,
            use_ocr=True,
            use_caption=False,
            use_detection=False,
            include_box_images=False,
        )
