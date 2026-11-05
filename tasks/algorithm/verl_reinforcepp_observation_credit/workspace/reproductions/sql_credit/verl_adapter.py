"""veRL's native stateful-tool interface; every rollout gets its own database."""
import json
import os
from pathlib import Path
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from reproductions.sql_credit.workload import Warehouse, grade, make_case


def case_from_id(case_id):
    split, index = case_id.rsplit("-", 1)
    return make_case(split, int(index))


class WarehouseTool(BaseTool):
    def __init__(self, config, tool_schema):
        self.instances = {}
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self):
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {"name": "warehouse", "description":
                "Inspect the warehouse schema or execute a read-only SQLite query.",
                "parameters": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["schema", "query"]},
                    "query": {"type": "string", "description": "SQL; use an empty string for schema."}},
                    "required": ["action", "query"], "additionalProperties": False}}})

    async def create(self, instance_id=None, case_id=None, **kwargs):
        case_id = case_id or kwargs.get("create_kwargs", {}).get("case_id")
        instance_id = instance_id or str(uuid4())
        self.instances[instance_id] = Warehouse(case_from_id(case_id))
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        env = self.instances[instance_id]
        result = env.call(parameters.get("action", ""), parameters.get("query", ""))
        root = Path(os.environ["PC_RUN_DIR"]) / "tool_traces"
        root.mkdir(exist_ok=True)
        with (root / f"{instance_id}.jsonl").open("a") as f:
            f.write(json.dumps({"case_id": env.case.case_id, "call": env.calls,
                                "parameters": parameters, "result": result}) + "\n")
        return ToolResponse(text=json.dumps(result)), 0.0, {"warehouse_calls": env.calls}

    async def release(self, instance_id, **kwargs):
        env = self.instances.pop(instance_id, None)
        if env:
            env.close()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    result = grade(case_from_id(ground_truth), solution_str)
    return {"score": result["score"], "acc": result["score"], "reason": result["reason"]}
