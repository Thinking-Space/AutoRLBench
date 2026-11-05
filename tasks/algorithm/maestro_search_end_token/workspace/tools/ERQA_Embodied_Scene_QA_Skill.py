import re

from tools.ERQA_Action_Adjustment_Solver import ERQA_Action_Adjustment_Solver_tool
from tools.ERQA_Multi_View_Task_Solver import ERQA_Multi_View_Task_Solver_tool
from tools.ERQA_Pointing_Localization_Solver import ERQA_Pointing_Localization_Solver_tool
from tools.ERQA_Spatial_Mechanics_Solver import ERQA_Spatial_Mechanics_Solver_tool
from tools.ERQA_Trajectory_Outcome_Solver import ERQA_Trajectory_Outcome_Solver_tool


DEFAULT_SYSTEM_PROMPT = (
    "You solve embodied scene question answering tasks by first recognizing the question type, "
    "then routing the question to the most suitable ERQA specialist solver."
)


ROUTE_TO_SOLVER = {
    "trajectory": ERQA_Trajectory_Outcome_Solver_tool,
    "action": ERQA_Action_Adjustment_Solver_tool,
    "spatial": ERQA_Spatial_Mechanics_Solver_tool,
    "pointing": ERQA_Pointing_Localization_Solver_tool,
    "multi_view": ERQA_Multi_View_Task_Solver_tool,
}

ROUTE_TO_NAME = {
    "trajectory": "ERQA_Trajectory_Outcome_Solver",
    "action": "ERQA_Action_Adjustment_Solver",
    "spatial": "ERQA_Spatial_Mechanics_Solver",
    "pointing": "ERQA_Pointing_Localization_Solver",
    "multi_view": "ERQA_Multi_View_Task_Solver",
}

ROUTE_PRIORITY = ["trajectory", "pointing", "multi_view", "action", "spatial"]

ROUTE_KEYWORDS = {
    "trajectory": [
        "trajectory",
        "path",
        "follow the line",
        "follow the trajectory",
        "follow the arrow",
        "along the arrow",
        "along the path",
        "move along",
        "colored path",
        "colored trajectory",
        "trace",
    ],
    "action": [
        "adjust",
        "align",
        "correction",
        "corrective",
        "move left",
        "move right",
        "move up",
        "move down",
        "rotate",
        "clockwise",
        "counterclockwise",
        "gripper",
        "end effector",
        "insert",
        "fit back",
        "reposition",
        "next action",
        "smallest motion",
    ],
    "spatial": [
        "left",
        "right",
        "front",
        "behind",
        "back",
        "top",
        "bottom",
        "inside",
        "outside",
        "between",
        "nearest",
        "closest",
        "farthest",
        "relative position",
        "hinge",
        "handle",
        "drawer",
        "knob",
        "mechanism",
        "open",
        "closed",
    ],
    "pointing": [
        "point",
        "dot",
        "coordinate",
        "labeled",
        "label",
        "marker",
        "arrow tip",
        "which location",
        "which part",
        "edge",
        "surface",
        "corner",
        "pixel",
    ],
    "multi_view": [
        "image 1",
        "image 2",
        "image 3",
        "image 4",
        "first image",
        "second image",
        "third image",
        "fourth image",
        "same point",
        "same part",
        "same corner",
        "same object",
        "same handle",
        "correspond",
        "corresponding",
        "other image",
        "another view",
        "multi-view",
        "multiple images",
        "which image",
        "closest to completing",
        "task state",
        "task completion",
    ],
}


def _normalize_question(question):
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _count_matches(question, keywords):
    return sum(1 for keyword in keywords if keyword in question)


class ERQA_Embodied_Scene_QA_Skill_tool:
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
        self.question_type = ""

    def _classify_question_type(self):
        question = _normalize_question(self.question)
        image_count = len(self.images) if self.images else len(self.image_paths)
        scores = {route_name: 0 for route_name in ROUTE_TO_SOLVER}

        for route_name, keywords in ROUTE_KEYWORDS.items():
            scores[route_name] += _count_matches(question, keywords)

        if image_count >= 2:
            scores["multi_view"] += 2
        if any(token in question for token in ["same point", "same part", "same corner", "correspond"]):
            scores["pointing"] += 2
        if any(token in question for token in ["which image", "closest to completing", "task state"]):
            scores["multi_view"] += 2
        if any(token in question for token in ["trajectory", "follow", "path", "arrow"]):
            scores["trajectory"] += 2
        if any(token in question for token in ["move left", "move right", "move up", "move down", "rotate", "align"]):
            scores["action"] += 2

        if all(score == 0 for score in scores.values()):
            return "spatial"

        best_route = "spatial"
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
        self.selected_skill = ROUTE_TO_NAME[self.question_type]
        solver = self._build_solver(ROUTE_TO_SOLVER[self.question_type])
        # print(
        #     f"[ERQA_Embodied_Scene_QA_Skill] question_type={self.question_type}, "
        #     f"selected_skill={self.selected_skill}"
        # )
        return solver.generate(is_test=is_test)
