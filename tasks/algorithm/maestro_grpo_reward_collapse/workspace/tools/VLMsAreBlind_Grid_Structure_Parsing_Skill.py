from tools.vlmsareblind_skill_utils import BaseVLMsAreBlindSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You parse regular grid and table layouts. "
    "Count rows and columns from the visible structure rather than from text content. "
    "Return only the final row and column values in the format requested."
)


class VLMsAreBlind_Grid_Structure_Parsing_Skill_tool(BaseVLMsAreBlindSkillTool):
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
            skill_name="Grid_Structure_Parsing_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="grid structure parsing",
            requirements=[
                "Ignore the cell contents and focus on the grid lines and outer border.",
                "Count rows from horizontal partitions and columns from vertical partitions.",
                "Do not confuse line thickness with extra rows or columns.",
                "If helpful, cross-check that rows multiplied by columns matches the visible cell count.",
                "Return only the final row and column values in the format requested by the question.",
            ],
            focus_hint="Focus on the outer border and the internal horizontal and vertical dividers of the grid.",
            max_crop_images=1,
        )


Grid_Structure_Parsing_Skill_tool = VLMsAreBlind_Grid_Structure_Parsing_Skill_tool
