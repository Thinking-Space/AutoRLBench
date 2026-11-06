import re

from tools.CLEVR_Attribute_Comparison_Skill import CLEVR_Attribute_Comparison_Skill_tool
from tools.CLEVR_Attribute_Query_Skill import CLEVR_Attribute_Query_Skill_tool
from tools.CLEVR_Count_Comparison_Skill import CLEVR_Count_Comparison_Skill_tool
from tools.CLEVR_Counting_Skill import CLEVR_Counting_Skill_tool
from tools.CLEVR_Existence_Skill import CLEVR_Existence_Skill_tool


DEFAULT_SYSTEM_PROMPT = (
    "You solve CLEVR questions by first recognizing the compositional reasoning pattern, "
    "then routing the question to the most suitable CLEVR specialist solver."
)

PRIMARY_SKILL_NAME = "Compositional_Visual_Reasoning_Skill"

ROUTE_TO_SOLVER = {
    "attribute_query": CLEVR_Attribute_Query_Skill_tool,
    "attribute_comparison": CLEVR_Attribute_Comparison_Skill_tool,
    "counting": CLEVR_Counting_Skill_tool,
    "count_comparison": CLEVR_Count_Comparison_Skill_tool,
    "existence": CLEVR_Existence_Skill_tool,
}

ROUTE_TO_NAME = {
    "attribute_query": "CLEVR_Attribute_Query_Skill",
    "attribute_comparison": "CLEVR_Attribute_Comparison_Skill",
    "counting": "CLEVR_Counting_Skill",
    "count_comparison": "CLEVR_Count_Comparison_Skill",
    "existence": "CLEVR_Existence_Skill",
}

ROUTE_PRIORITY = [
    "count_comparison",
    "counting",
    "existence",
    "attribute_comparison",
    "attribute_query",
]

ROUTE_KEYWORDS = {
    "attribute_query": [
        "what color",
        "what is the color",
        "what material",
        "what is the material",
        "what shape",
        "what is the shape",
        "what size",
        "what is the size",
        "is what color",
        "is what shape",
        "is what size",
        "made of what",
    ],
    "attribute_comparison": [
        "same color as",
        "same material as",
        "same shape as",
        "same size as",
        "same color",
        "same material",
        "same shape",
        "same size",
        "made of the same material",
    ],
    "counting": [
        "what number of",
        "how many",
    ],
    "count_comparison": [
        "same number of",
        "same number as",
        "greater than",
        "less than",
        "fewer than",
        "more than",
        "are there more",
        "are there fewer",
    ],
    "existence": [
        "is there",
        "are there any",
        "is there any",
        "is there another",
        "have any other",
    ],
}


def _normalize_question(question):
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _count_matches(question, keywords):
    return sum(1 for keyword in keywords if keyword in question)


class CLEVR_Problem_Solver_tool:
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

        if (
            question.startswith("is there")
            or question.startswith("are there any")
            or question.startswith("is there any")
            or question.startswith("is there another")
        ):
            return "existence"
        if question.startswith("are there more") or question.startswith("are there fewer"):
            return "count_comparison"
        if "what number of" in question or "how many" in question:
            return "counting"

        scores = {route_name: 0 for route_name in ROUTE_TO_SOLVER}

        for route_name, keywords in ROUTE_KEYWORDS.items():
            scores[route_name] += _count_matches(question, keywords)

        if "number of" in question and any(
            token in question for token in ["same", "greater than", "less than", "fewer than", "more than"]
        ):
            scores["count_comparison"] += 3
        if any(
            token in question
            for token in ["same color as", "same material as", "same shape as", "same size as"]
        ):
            scores["attribute_comparison"] += 3
        if any(
            token in question
            for token in ["what color", "what material", "what shape", "what size", "is what color", "is what shape", "is what size"]
        ):
            scores["attribute_query"] += 3
        if any(token in question for token in ["is there", "are there any", "is there any", "have any other"]):
            scores["existence"] += 2

        if all(score == 0 for score in scores.values()):
            return "attribute_query"

        best_route = "attribute_query"
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
        # print(
        #     f"[{PRIMARY_SKILL_NAME}] question_type={self.question_type}, "
        #     f"selected_skill={self.selected_skill}, "
        #     f"selected_subskill={self.selected_subskill}"
        # )
        return solver.generate(is_test=is_test)


CLEVR_Master_Router_Skill_tool = CLEVR_Problem_Solver_tool
