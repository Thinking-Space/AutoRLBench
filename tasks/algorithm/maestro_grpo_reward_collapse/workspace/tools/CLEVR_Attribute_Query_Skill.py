from tools.clevr_skill_utils import BaseCLEVRSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve CLEVR attribute query questions. "
    "Resolve the referred object using all filters and spatial relations, "
    "then answer with only the requested attribute value."
)


class CLEVR_Attribute_Query_Skill_tool(BaseCLEVRSkillTool):
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
            skill_name="CLEVR_Attribute_Query_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="attribute query",
            requirements=[
                "Identify the exact target object described by the question before answering.",
                "Apply color, material, shape, size, and spatial filters together.",
                "Answer with only the requested attribute value, such as a color, material, shape, or size word.",
                "Do not explain your reasoning. The final line must be only the answer.",
            ],
            focus_hint="Focus on the specifically referenced CLEVR object and the nearby objects needed to resolve left, right, front, or behind relations.",
            max_crop_images=1,
        )
