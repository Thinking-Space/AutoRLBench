from enum import Enum

# Import tools directly from current directory (tools/)
from tools.self_consistancy import Self_consistency_tool
from tools.none import None_tool
# from tools.web_search import Web_search_tool
# from tools.python_code import Python_code_tool
# from tools.calculator import Calculator_tool
# from tools.prm import Prm_tool
from tools.ocr import Ocr_tool
from tools.yolo import Yolo_tool
from tools.detection import Detection_tool
from tools.imagecaption import ImageCaption_tool
from tools.Geometric_Problem_Solver import Geometric_Problem_Solver_tool
from tools.Chart_Problem_Solver import Chart_Problem_Solver_tool
from tools.Science_Problem_Solver import Science_Problem_Solver_tool
from tools.Counting_Problem_Solver import Counting_Problem_Solver_tool
from tools.Perception_Problem_Solver import Perception_Problem_Solver_tool
from tools.Perception_Problem_Solver_Eval import Perception_Problem_Solver_Eval_tool
from tools.Code_Problem_Solver import Code_Problem_Solver_tool
from tools.zoom import Zoom_tool
from tools.ERQA_Trajectory_Outcome_Solver import ERQA_Trajectory_Outcome_Solver_tool
from tools.ERQA_Action_Adjustment_Solver import ERQA_Action_Adjustment_Solver_tool
from tools.ERQA_Spatial_Mechanics_Solver import ERQA_Spatial_Mechanics_Solver_tool
from tools.ERQA_Pointing_Localization_Solver import ERQA_Pointing_Localization_Solver_tool
from tools.ERQA_Multi_View_Task_Solver import ERQA_Multi_View_Task_Solver_tool
from tools.ERQA_Embodied_Scene_QA_Skill import ERQA_Embodied_Scene_QA_Skill_tool
from tools.CLEVR_Master_Router_Skill import CLEVR_Master_Router_Skill_tool
from tools.OCRBench_Text_Recognition_Skill import OCRBench_Text_Recognition_Skill_tool
from tools.OCRBench_Key_Information_Extraction_Skill import OCRBench_Key_Information_Extraction_Skill_tool
from tools.OCRBench_Scene_Text_QA_Skill import OCRBench_Scene_Text_QA_Skill_tool
from tools.OCRBench_Document_Chart_QA_Skill import OCRBench_Document_Chart_QA_Skill_tool
from tools.OCRBench_Formula_Recognition_Skill import OCRBench_Formula_Recognition_Skill_tool
from tools.OCRBench_Master_Router_Skill import OCRBench_Master_Router_Skill_tool
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
from tools.VLMsAreBlind_Master_Router_Skill import (
    VLMsAreBlind_Master_Router_Skill_tool,
)
    
class ToolFactory:
    def __init__(self, client, tool_client, tool_model_name, model, system_prompt, question, images, have_image, temperature, top_p, timeout, image_paths=[], dataset=""):
        self.client = client
        self.model = model
        self.tool_client = tool_client
        self.tool_model_name = tool_model_name
        # self.messages = messages
        self.system_prompt = system_prompt
        self.question = question
        self.images = images
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image
        self.image_paths = image_paths
        self.dataset = dataset
    
    def get_tool(self, tool):
        # import pdb; pdb.set_trace()
        if tool.tool_name == "self-consistency":
            return Self_consistency_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image)
        elif tool.tool_name == "none":
            return None_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image)
        # elif tool.tool_name == "web-search":
        #     return Web_search_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image)
        # elif tool.tool_name == "python-code":
        #     return Python_code_tool(tool = tool, tool_model_name = self.tool_model_name, code_client = self.tool_client, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image)
        # elif tool.tool_name == "calculator":
        #     return Calculator_tool(tool = tool, tool_model_name = self.tool_model_name, tool_client = self.tool_client, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image)
        # elif tool.tool_name == "prm":
        #     return Prm_tool(tool = tool, tool_client = self.tool_client, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image)
        elif tool.tool_name == "ocr":
            return Ocr_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name == "yolo":
            return Yolo_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name == "detection":
            return Detection_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name == "imagecaption":
            return ImageCaption_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"Geometric_Problem_Solver", "geometric_problem_solver", "geometric-problem-solver"}:
            return Geometric_Problem_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"Chart_Problem_Solver", "chart_problem_solver", "chart-problem-solver"}:
            return Chart_Problem_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"Science_Problem_Solver", "science_problem_solver", "science-problem-solver"}:
            return Science_Problem_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"Counting_Problem_Solver", "counting_problem_solver", "counting-problem-solver"}:
            return Counting_Problem_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"Perception_Problem_Solver", "perception_problem_solver", "perception-problem-solver"}:
            return Perception_Problem_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"Perception_Problem_Solver_Eval", "perception_problem_solver_eval", "perception-problem-solver-eval"}:
            return Perception_Problem_Solver_Eval_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"Code_Problem_Solver", "code_problem_solver", "code-problem-solver"}:
            return Code_Problem_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"ERQA_Trajectory_Outcome_Solver", "erqa_trajectory_outcome_solver", "erqa-trajectory-outcome-solver"}:
            return ERQA_Trajectory_Outcome_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"ERQA_Action_Adjustment_Solver", "erqa_action_adjustment_solver", "erqa-action-adjustment-solver"}:
            return ERQA_Action_Adjustment_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"ERQA_Spatial_Mechanics_Solver", "erqa_spatial_mechanics_solver", "erqa-spatial-mechanics-solver"}:
            return ERQA_Spatial_Mechanics_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"ERQA_Pointing_Localization_Solver", "erqa_pointing_localization_solver", "erqa-pointing-localization-solver"}:
            return ERQA_Pointing_Localization_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"ERQA_Multi_View_Task_Solver", "erqa_multi_view_task_solver", "erqa-multi-view-task-solver"}:
            return ERQA_Multi_View_Task_Solver_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Embodied_Scene_QA_Skill",
            "embodied_scene_qa_skill",
            "embodied-scene-qa-skill",
            "ERQA_Embodied_Scene_QA_Skill",
            "erqa_embodied_scene_qa_skill",
            "erqa-embodied-scene-qa-skill",
        }:
            return ERQA_Embodied_Scene_QA_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Compositional_Visual_Reasoning_Skill",
            "compositional_visual_reasoning_skill",
            "compositional-visual-reasoning-skill",
            "clevr_problem_solver",
            "clevr-problem-solver",
            "CLEVR_Problem_Solver",
            "CLEVR_Master_Router_Skill",
            "clevr_master_router_skill",
            "clevr-master-router-skill",
        }:
            return CLEVR_Master_Router_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"OCRBench_Text_Recognition_Skill", "ocrbench_text_recognition_skill", "ocrbench-text-recognition-skill"}:
            return OCRBench_Text_Recognition_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"OCRBench_Key_Information_Extraction_Skill", "ocrbench_key_information_extraction_skill", "ocrbench-key-information-extraction-skill"}:
            return OCRBench_Key_Information_Extraction_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"OCRBench_Scene_Text_QA_Skill", "ocrbench_scene_text_qa_skill", "ocrbench-scene-text-qa-skill"}:
            return OCRBench_Scene_Text_QA_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"OCRBench_Document_Chart_QA_Skill", "ocrbench_document_chart_qa_skill", "ocrbench-document-chart-qa-skill"}:
            return OCRBench_Document_Chart_QA_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {"OCRBench_Formula_Recognition_Skill", "ocrbench_formula_recognition_skill", "ocrbench-formula-recognition-skill"}:
            return OCRBench_Formula_Recognition_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "ocr_problem_solver",
            "ocr-problem-solver",
            "OCR_Problem_Solver",
            "OCRBench_Master_Router_Skill",
            "ocrbench_master_router_skill",
            "ocrbench-master-router-skill",
        }:
            return OCRBench_Master_Router_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Circle_Contact_and_Overlap_Judge_Skill",
            "circle_contact_and_overlap_judge_skill",
            "circle-contact-and-overlap-judge-skill",
            "VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill",
            "vlmsareblind_circle_contact_overlap_judge_skill",
            "vlmsareblind-circle-contact-overlap-judge-skill",
        }:
            return VLMsAreBlind_Circle_Contact_Overlap_Judge_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Intersection_and_Route_Counting_Skill",
            "intersection_and_route_counting_skill",
            "intersection-and-route-counting-skill",
            "VLMsAreBlind_Intersection_Route_Counting_Skill",
            "vlmsareblind_intersection_route_counting_skill",
            "vlmsareblind-intersection-route-counting-skill",
        }:
            return VLMsAreBlind_Intersection_Route_Counting_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Grid_Structure_Parsing_Skill",
            "grid_structure_parsing_skill",
            "grid-structure-parsing-skill",
            "VLMsAreBlind_Grid_Structure_Parsing_Skill",
            "vlmsareblind_grid_structure_parsing_skill",
            "vlmsareblind-grid-structure-parsing-skill",
        }:
            return VLMsAreBlind_Grid_Structure_Parsing_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Highlighted_Character_Recognition_Skill",
            "highlighted_character_recognition_skill",
            "highlighted-character-recognition-skill",
            "VLMsAreBlind_Highlighted_Character_Recognition_Skill",
            "vlmsareblind_highlighted_character_recognition_skill",
            "vlmsareblind-highlighted-character-recognition-skill",
        }:
            return VLMsAreBlind_Highlighted_Character_Recognition_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Geometric_Shape_Counting_Skill",
            "geometric_shape_counting_skill",
            "geometric-shape-counting-skill",
            "VLMsAreBlind_Geometric_Shape_Counting_Skill",
            "vlmsareblind_geometric_shape_counting_skill",
            "vlmsareblind-geometric-shape-counting-skill",
        }:
            return VLMsAreBlind_Geometric_Shape_Counting_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name in {
            "Diagram_Reasoning_Skill",
            "diagram_reasoning_skill",
            "diagram-reasoning-skill",
            "Diagram_Reasoning_Router_Skill",
            "diagram_reasoning_router_skill",
            "diagram-reasoning-router-skill",
            "VLMsAreBlind_Master_Router_Skill",
            "vlmsareblind_master_router_skill",
            "vlmsareblind-master-router-skill",
            "vlmsareblind_problem_solver",
            "vlmsareblind-problem-solver",
        }:
            return VLMsAreBlind_Master_Router_Skill_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        elif tool.tool_name == "zoom":
            return Zoom_tool(tool = tool, client = self.client, model = self.model, system_prompt = self.system_prompt, question = self.question, images = self.images, temperature = self.temperature, top_p = self.top_p, timeout = self.timeout, have_image = self.have_image, image_paths=self.image_paths, dataset=self.dataset)
        else:
            raise ValueError(f"Invalid tool: {tool.tool_name}")
        
