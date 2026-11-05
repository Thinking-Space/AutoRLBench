from tools.vlmsareblind_skill_utils import BaseVLMsAreBlindSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You judge whether two circles are touching or overlapping. "
    "Distinguish separation, tangency, and area overlap carefully. "
    "Return only the final Yes/No answer."
)


class VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill_tool(BaseVLMsAreBlindSkillTool):
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
            skill_name="Circle_Contact_and_Overlap_Judge_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="circle contact and overlap judgment",
            requirements=[
                "Focus only on the two target circles.",
                "Decide whether the circles are separated, exactly touching, or overlapping with shared area.",
                "If the question asks about touching, answer based on boundary contact only.",
                "If the question asks about overlapping, answer based on shared area only.",
                "Output only Yes or No with no explanation.",
            ],
            focus_hint="Focus tightly on the gap or overlap region between the two circles.",
            max_crop_images=1,
        )


Circle_Contact_and_Overlap_Judge_Skill_tool = (
    VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill_tool
)
