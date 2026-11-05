import base64
import os
import re

from tools.OCRBench_Document_Chart_QA_Skill import OCRBench_Document_Chart_QA_Skill_tool
from tools.OCRBench_Formula_Recognition_Skill import OCRBench_Formula_Recognition_Skill_tool
from tools.OCRBench_Key_Information_Extraction_Skill import (
    OCRBench_Key_Information_Extraction_Skill_tool,
)
from tools.OCRBench_Scene_Text_QA_Skill import OCRBench_Scene_Text_QA_Skill_tool
from tools.OCRBench_Text_Recognition_Skill import OCRBench_Text_Recognition_Skill_tool


DEFAULT_SYSTEM_PROMPT = (
    "You solve OCRBench tasks by first recognizing the OCR task type, "
    "then routing the question to the most suitable OCR specialist solver."
)

PRIMARY_SKILL_NAME = "ocr_problem_solver"

ROUTE_TO_SOLVER = {
    "text_recognition": OCRBench_Text_Recognition_Skill_tool,
    "key_information_extraction": OCRBench_Key_Information_Extraction_Skill_tool,
    "scene_text_qa": OCRBench_Scene_Text_QA_Skill_tool,
    "document_chart_qa": OCRBench_Document_Chart_QA_Skill_tool,
    "formula_recognition": OCRBench_Formula_Recognition_Skill_tool,
}

ROUTE_TO_NAME = {
    "text_recognition": "OCRBench_Text_Recognition_Skill",
    "key_information_extraction": "OCRBench_Key_Information_Extraction_Skill",
    "scene_text_qa": "OCRBench_Scene_Text_QA_Skill",
    "document_chart_qa": "OCRBench_Document_Chart_QA_Skill",
    "formula_recognition": "OCRBench_Formula_Recognition_Skill",
}

ROUTE_PRIORITY = [
    "formula_recognition",
    "key_information_extraction",
    "document_chart_qa",
    "scene_text_qa",
    "text_recognition",
]

ROUTE_KEYWORDS = {
    "formula_recognition": [
        "latex format",
        "expression of the formula",
        "formula in the image",
        "equation",
        "expression",
    ],
    "key_information_extraction": [
        "total amount",
        "receipt",
        "issued",
        "company",
        "per 100g/ml",
        "per serving",
        "value for",
        "nutrition",
        "protein",
        "sodium",
        "calories",
        "total fat",
        "invoice",
        "address",
        "date",
    ],
    "document_chart_qa": [
        "chart",
        "table",
        "graph",
        "calendar",
        "book",
        "author",
        "title",
        "year",
        "percentage",
        "export value",
        "national debt",
        "million",
        "population",
        "country",
    ],
    "scene_text_qa": [
        "boat",
        "shirt",
        "store",
        "sign",
        "street",
        "shop",
        "white illuminated letters",
        "on the woman's shirt",
        "on the man's shirt",
        "name of this",
        "what store",
        "what is written on",
    ],
    "text_recognition": [
        "what is written in the image",
        "what is the number in the image",
    ],
}


def _normalize_question(question):
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _count_matches(question, keywords):
    return sum(1 for keyword in keywords if keyword in question)


class OCR_Problem_Solver_tool:
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

        if "latex" in question or "formula" in question:
            scores["formula_recognition"] += 3
        if "what is written in the image" in question or "what is the number in the image" in question:
            scores["text_recognition"] += 3
        if any(
            token in question
            for token in ["total amount", "per 100g/ml", "per serving", "issued", "company"]
        ):
            scores["key_information_extraction"] += 3
        if any(
            token in question
            for token in ["book", "author", "title", "calendar", "chart", "graph", "table"]
        ):
            scores["document_chart_qa"] += 2
        if any(
            token in question
            for token in ["shirt", "boat", "store", "sign", "written on", "name of this"]
        ):
            scores["scene_text_qa"] += 2

        if all(score == 0 for score in scores.values()):
            return "scene_text_qa"

        best_route = "scene_text_qa"
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


OCRBench_Master_Router_Skill_tool = OCR_Problem_Solver_tool


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


