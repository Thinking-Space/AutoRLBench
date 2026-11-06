from tools.clevr_skill_utils import BaseCLEVRSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You solve CLEVR counting questions. "
    "Count exactly the objects that satisfy the full description in the question "
    "and output only the final number."
)


class CLEVR_Counting_Skill_tool(BaseCLEVRSkillTool):
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
            skill_name="CLEVR_Counting_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="counting",
            requirements=[
                "Count only the objects that satisfy every stated attribute and relation in the question.",
                "Do not include the reference object itself unless the wording explicitly allows it.",
                "Return the final count as only a number.",
                "Do not explain your reasoning.",
            ],
            focus_hint="Focus on the full set of CLEVR objects that satisfy the filters and relations in the question, and separate anchors from counted targets.",
            max_crop_images=1,
        )
