from tools.vlmsareblind_skill_utils import BaseVLMsAreBlindSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You identify the single highlighted character in a string of letters. "
    "Localize the highlight first, then read only the marked character, preserving case. "
    "Return only the final character in the format requested."
)


class VLMsAreBlind_Highlighted_Character_Recognition_Skill_tool(
    BaseVLMsAreBlindSkillTool
):
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
            skill_name="Highlighted_Character_Recognition_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="highlighted character recognition",
            requirements=[
                "Localize the circle or red oval before reading any character.",
                "Read only the single character at the center of the highlight.",
                "Preserve uppercase and lowercase exactly.",
                "Ignore nearby characters outside the highlight region.",
                "Return only the final character, with braces only if the question asks for braces.",
            ],
            focus_hint="Focus on the red circle or oval and the single character centered inside it.",
            max_crop_images=1,
            use_ocr=True,
        )


Highlighted_Character_Recognition_Skill_tool = (
    VLMsAreBlind_Highlighted_Character_Recognition_Skill_tool
)
