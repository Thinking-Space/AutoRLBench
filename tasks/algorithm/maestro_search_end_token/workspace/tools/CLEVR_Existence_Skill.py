from tools.clevr_skill_utils import BaseCLEVRSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve CLEVR existence questions. "
    "Determine whether at least one object satisfies the full description "
    "and answer with yes or no."
)


class CLEVR_Existence_Skill_tool(BaseCLEVRSkillTool):
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
            skill_name="CLEVR_Existence_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="existence verification",
            requirements=[
                "Check whether there exists at least one object that satisfies the full chain of filters and relations.",
                "Be careful with descriptions involving another object that shares the same attribute as a reference object.",
                "Answer only yes or no.",
                "Do not explain your reasoning.",
            ],
            focus_hint="Focus on candidate CLEVR objects that could satisfy the full description, and verify each required attribute and relation before deciding whether any valid object exists.",
            max_crop_images=1,
        )
