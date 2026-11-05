import re

from tools.VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill import (
    VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill_tool,
)
from tools.VLMsAreBlind_Geometric_Shape_Counting_Skill import (
    VLMsAreBlind_Geometric_Shape_Counting_Skill_tool,
)
from tools.VLMsAreBlind_Grid_Structure_Parsing_Skill import (
    VLMsAreBlind_Grid_Structure_Parsing_Skill_tool,
)
from tools.VLMsAreBlind_Highlighted_Character_Recognition_Skill import (
    VLMsAreBlind_Highlighted_Character_Recognition_Skill_tool,
)
from tools.VLMsAreBlind_Intersection_Route_Counting_Skill import (
    VLMsAreBlind_Intersection_Route_Counting_Skill_tool,
)


DEFAULT_SYSTEM_PROMPT = (
    "You route synthetic diagram reasoning questions to the most suitable specialist solver "
    "using simple keyword matching on the question."
)

PRIMARY_SKILL_NAME = "Diagram_Reasoning_Skill"

ROUTE_TO_SOLVER = {
    "contact_overlap": VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill_tool,
    "intersection_route_counting": VLMsAreBlind_Intersection_Route_Counting_Skill_tool,
    "grid_structure_parsing": VLMsAreBlind_Grid_Structure_Parsing_Skill_tool,
    "highlighted_character_recognition": (
        VLMsAreBlind_Highlighted_Character_Recognition_Skill_tool
    ),
    "geometric_shape_counting": VLMsAreBlind_Geometric_Shape_Counting_Skill_tool,
}

ROUTE_TO_NAME = {
    "contact_overlap": "Circle_Contact_and_Overlap_Judge_Skill",
    "intersection_route_counting": "Intersection_and_Route_Counting_Skill",
    "grid_structure_parsing": "Grid_Structure_Parsing_Skill",
    "highlighted_character_recognition": "Highlighted_Character_Recognition_Skill",
    "geometric_shape_counting": "Geometric_Shape_Counting_Skill",
}

ROUTE_PRIORITY = [
    "highlighted_character_recognition",
    "grid_structure_parsing",
    "contact_overlap",
    "intersection_route_counting",
    "geometric_shape_counting",
]

ROUTE_KEYWORDS = {
    "contact_overlap": [
        "overlapping",
        "touching each other",
        "two circles",
    ],
    "intersection_route_counting": [
        "blue and red lines",
        "intersection points",
        "touch each other",
        "single-color paths",
        "single-color path",
        "one-colored routes",
        "one-colored route",
        "go from ",
        "routes that go from",
        "paths go from",
    ],
    "grid_structure_parsing": [
        "rows and columns",
        "row,column",
        "table",
        "rows={",
        "columns={",
    ],
    "highlighted_character_recognition": [
        "which letter is being circled",
        "which character is being highlighted",
        "red oval",
        "being circled",
        "being highlighted",
    ],
    "geometric_shape_counting": [
        "how many circles",
        "count the circles",
        "count the pentagons",
        "how many pentagons",
        "how many squares",
        "count total number of squares",
        "count the number of squares",
    ],
}


def _normalize_question(question):
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _count_matches(question, keywords):
    return sum(1 for keyword in keywords if keyword in question)


class VLMsAreBlind_Problem_Solver_tool:
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
        self.tool = tool
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.question = question
        self.images = images
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image
        self.image_paths = image_paths if image_paths else []
        self.dataset = dataset
        self.selected_skill = ""
        self.selected_subskill = ""
        self.question_type = ""

    def _classify_question_type(self):
        question = _normalize_question(self.question)
        scores = {route_name: 0 for route_name in ROUTE_TO_SOLVER}

        for route_name, keywords in ROUTE_KEYWORDS.items():
            scores[route_name] += _count_matches(question, keywords)

        if "which letter" in question or "which character" in question:
            scores["highlighted_character_recognition"] += 3
        if "rows and columns" in question or "(row,column)" in question:
            scores["grid_structure_parsing"] += 3
        if "overlapping" in question or "touching each other" in question:
            scores["contact_overlap"] += 3
        if "intersection points" in question or "blue and red lines" in question:
            scores["intersection_route_counting"] += 3
        if "single-color" in question or "one-colored" in question:
            scores["intersection_route_counting"] += 3
        if any(
            token in question
            for token in [
                "how many circles",
                "count the pentagons",
                "how many pentagons",
                "how many squares",
                "count total number of squares",
            ]
        ):
            scores["geometric_shape_counting"] += 3

        if all(score == 0 for score in scores.values()):
            return "geometric_shape_counting"

        best_route = "geometric_shape_counting"
        best_score = -1
        for route_name in ROUTE_PRIORITY:
            score = scores[route_name]
            if score > best_score:
                best_route = route_name
                best_score = score
        return best_route

    def _build_solver(self, solver_cls):
        return solver_cls(
            tool=self.tool,
            client=self.client,
            model=self.model,
            system_prompt=self.system_prompt or DEFAULT_SYSTEM_PROMPT,
            question=self.question,
            images=self.images,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            have_image=self.have_image,
            image_paths=self.image_paths,
            dataset=self.dataset,
        )

    def generate(self, is_test=False):
        if not self.images:
            return None

        self.question_type = self._classify_question_type()
        self.selected_skill = PRIMARY_SKILL_NAME
        self.selected_subskill = ROUTE_TO_NAME[self.question_type]
        solver = self._build_solver(ROUTE_TO_SOLVER[self.question_type])
        return solver.generate(is_test=is_test)


VLMsAreBlind_Master_Router_Skill_tool = VLMsAreBlind_Problem_Solver_tool
Diagram_Reasoning_Skill_tool = VLMsAreBlind_Problem_Solver_tool
Diagram_Reasoning_Router_Skill_tool = VLMsAreBlind_Problem_Solver_tool
import base64
import os
import re

def _encode_image_to_data_url(image_path):
    extension = os.path.splitext(image_path)[1].lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(extension, "image/jpeg")

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


