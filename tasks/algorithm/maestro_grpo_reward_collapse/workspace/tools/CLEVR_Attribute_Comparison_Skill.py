from tools.clevr_skill_utils import BaseCLEVRSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve CLEVR attribute comparison questions. "
    "Resolve both referenced objects carefully and compare only the requested attribute. "
    "Answer with yes or no."
)


class CLEVR_Attribute_Comparison_Skill_tool(BaseCLEVRSkillTool):
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
            skill_name="CLEVR_Attribute_Comparison_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="attribute comparison",
            requirements=[
                "Resolve the first referenced object and the second referenced object separately before comparing them.",
                "Compare only the asked attribute, such as color, material, shape, or size.",
                "Be careful when each object is defined through a spatial relation to another object.",
                "Answer only yes or no, with no explanation.",
            ],
            focus_hint="Focus on the two described CLEVR objects and any anchor objects needed to resolve the reference chain before comparing the requested attribute.",
            max_crop_images=1,
        )
