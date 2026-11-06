from tools.ocrbench_skill_utils import BaseOCRBenchSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve OCRBench key information extraction tasks. "
    "Locate the requested field in the document and return only its value."
)


class OCRBench_Key_Information_Extraction_Skill_tool(BaseOCRBenchSkillTool):
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
            skill_name="OCRBench_Key_Information_Extraction_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="key information extraction",
            requirements=[
                "Identify the requested field type before reading the page.",
                "Use layout cues to distinguish field names from nearby values.",
                "Prefer the exact field value requested by the question, including units when present.",
                "Do not explain the answer. The final line must be only the extracted value.",
            ],
            focus_hint="Focus on receipts, forms, tables, field labels, totals, dates, company names, and nutrition-value regions that match the requested field.",
            max_crop_images=1,
            use_ocr=True,
            use_caption=False,
            use_detection=True,
            include_box_images=True,
        )
