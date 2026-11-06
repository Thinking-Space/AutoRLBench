from tools.vlmsareblind_skill_utils import BaseVLMsAreBlindSkillTool


DEFAULT_SYSTEM_PROMPT = (
    "You count line intersections and single-color routes in synthetic diagrams. "
    "Treat the diagram as a discrete structure and count only valid crossings or valid routes. "
    "Return only the final answer in the format requested by the question."
)


class VLMsAreBlind_Intersection_Route_Counting_Skill_tool(BaseVLMsAreBlindSkillTool):
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
            skill_name="Intersection_and_Route_Counting_Skill",
            default_system_prompt=DEFAULT_SYSTEM_PROMPT,
            task_intro="intersection counting and single-color route counting",
            requirements=[
                "First determine whether the question is about line intersections or colored routes.",
                "For line intersections, count only true red-blue meeting points.",
                "Do not count near misses, parallel closeness, or visual thickness artifacts as intersections.",
                "For route questions, count only complete start-to-end paths that stay within one color.",
                "Return only the final number, using braces only if the question requests braces.",
            ],
            focus_hint=(
                "Focus on red-blue crossing points, or on the labeled endpoints and the colored route segments "
                "that connect them without changing color."
            ),
            max_crop_images=1,
        )


Intersection_and_Route_Counting_Skill_tool = (
    VLMsAreBlind_Intersection_Route_Counting_Skill_tool
)
