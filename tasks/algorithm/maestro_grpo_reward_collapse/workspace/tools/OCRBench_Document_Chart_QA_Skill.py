from tools.ocrbench_skill_utils import BaseOCRBenchSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve OCRBench document and chart question answering tasks. "
    "Read structured pages carefully and answer using the relevant text, layout, or chart values."
)


class OCRBench_Document_Chart_QA_Skill_tool(BaseOCRBenchSkillTool):
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
            skill_name="OCRBench_Document_Chart_QA_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="document and chart question answering",
            requirements=[
                "Determine whether the page is a document, infographic, table, chart, or calendar before answering.",
                "Use OCR and page structure jointly rather than relying on raw text alone.",
                "For charts, match labels, axes, legends, and values carefully before selecting the answer.",
                "Do not explain the answer. The final line must be only the requested answer text or value.",
            ],
            focus_hint="Focus on titles, headers, table cells, chart labels, axes, legends, dates, and the local region that directly supports the question.",
            max_crop_images=1,
            use_ocr=True,
            use_caption=True,
            use_detection=False,
            include_box_images=False,
        )
