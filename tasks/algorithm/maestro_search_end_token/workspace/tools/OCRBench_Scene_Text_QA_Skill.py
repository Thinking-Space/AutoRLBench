from tools.ocrbench_skill_utils import BaseOCRBenchSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve OCRBench scene text question answering tasks. "
    "First find the correct object in the scene, then read the text attached to it."
)


class OCRBench_Scene_Text_QA_Skill_tool(BaseOCRBenchSkillTool):
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
            skill_name="OCRBench_Scene_Text_QA_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="scene text question answering",
            requirements=[
                "Identify which real-world object the question refers to before reading text.",
                "Separate relevant text on the target object from unrelated background text.",
                "Use OCR as supporting evidence, but rely on the image to resolve which text belongs to the target.",
                "Do not explain the answer. The final line must be only the requested answer text.",
            ],
            focus_hint="Focus on the object mentioned in the question, such as a sign, shirt, boat, storefront, label, package, or board, and crop the text attached to that object.",
            max_crop_images=1,
            use_ocr=True,
            use_caption=True,
            use_detection=True,
            include_box_images=True,
        )
