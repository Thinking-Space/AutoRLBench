from tools.ocrbench_skill_utils import BaseOCRBenchSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve OCRBench handwritten mathematical expression recognition tasks. "
    "Recover the two-dimensional math structure and output only valid LaTeX."
)


class OCRBench_Formula_Recognition_Skill_tool(BaseOCRBenchSkillTool):
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
            skill_name="OCRBench_Formula_Recognition_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="formula recognition",
            requirements=[
                "Recover mathematical structure rather than only reading symbols one by one.",
                "Pay special attention to fractions, superscripts, subscripts, brackets, roots, and Greek letters.",
                "Output the expression in LaTeX format only.",
                "Do not explain the answer. The final line must be only the LaTeX expression.",
            ],
            focus_hint="Focus tightly on the handwritten formula, including fraction bars, exponents, subscripts, radicals, equality signs, and grouped terms.",
            max_crop_images=1,
            use_ocr=True,
            use_caption=False,
            use_detection=False,
            include_box_images=False,
        )
