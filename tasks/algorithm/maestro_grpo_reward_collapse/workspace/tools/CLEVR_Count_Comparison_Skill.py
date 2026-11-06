from tools.clevr_skill_utils import BaseCLEVRSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve CLEVR count comparison questions. "
    "Count the two referenced object sets accurately and answer yes or no "
    "according to the requested comparison."
)


class CLEVR_Count_Comparison_Skill_tool(BaseCLEVRSkillTool):
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
            skill_name="CLEVR_Count_Comparison_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="count comparison",
            requirements=[
                "Identify the first counted set and the second counted set separately.",
                "Apply all attributes and spatial relations precisely before counting either set.",
                "Compare the two final counts according to words such as same, more, fewer, greater, or less.",
                "Answer only yes or no, with no explanation.",
            ],
            focus_hint="Focus on the two counted CLEVR object groups, especially when each group is defined by a different relation or reference object.",
            max_crop_images=1,
        )
