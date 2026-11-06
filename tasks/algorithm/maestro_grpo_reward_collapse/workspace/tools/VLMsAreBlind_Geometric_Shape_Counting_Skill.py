from tools.vlmsareblind_skill_utils import BaseVLMsAreBlindSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You count repeated geometric shapes in synthetic diagrams. "
    "Identify the target shape type first, then count complete instances carefully without duplicates. "
    "Return only the final number in the format requested."
)


class VLMsAreBlind_Geometric_Shape_Counting_Skill_tool(BaseVLMsAreBlindSkillTool):
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
            skill_name="Geometric_Shape_Counting_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="geometric shape counting",
            requirements=[
                "Identify exactly which shape type the question asks for before counting.",
                "Count only complete target shapes and avoid double counting overlapping outlines.",
                "For nested squares, count closed square contours rather than corners or edge segments.",
                "Use a stable scan order to avoid misses or duplicates.",
                "Return only the final number, using braces only if the question asks for braces.",
            ],
            focus_hint="Focus on complete circles, pentagons, or square outlines and scan them in a stable order.",
            max_crop_images=1,
        )


Geometric_Shape_Counting_Skill_tool = (
    VLMsAreBlind_Geometric_Shape_Counting_Skill_tool
)
