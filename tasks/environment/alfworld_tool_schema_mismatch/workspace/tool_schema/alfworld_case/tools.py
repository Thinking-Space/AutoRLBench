"""Execution registry and the tool schema advertised to the model."""
import copy
import json


SYSTEM_PROMPT = """You are an agent interacting with a text-based household environment.
Complete the user's task using the provided tools.

Call exactly one tool at each turn.
Use object and location identifiers as they appear in observations.
Explore the environment to obtain missing information.
The initial observation lists places you can visit. Move to a place to inspect its contents.
Objects in closed containers are hidden until the container is opened at its location.
Looking only describes the current location; it does not search other places in the room.
When an observation is unchanged, explore a different place or try a different action.
Use tool results to track the environment and decide your next action.
If a call fails, use the feedback to decide how to proceed.
Do not claim task completion; success is determined by the environment."""

# Native ALFWorld uses "move ... to ..." for placement, not "put ... in ...".
REGISTRY = {
    "look": ("Observe only your current location, not the whole room. At the room center it may show nothing; move to a listed location to explore.", (), "look"),
    "inventory": ("Inspect the objects you are holding.", (), "inventory"),
    "go_to": ("Move to a receptacle or location listed in observations and see its visible contents. Use this to explore, not the name of a small object.", ("destination",), "go to {destination}"),
    "open": ("Open a container at your current location to reveal its contents.", ("target",), "open {target}"),
    "close": ("Close a container at your current location.", ("target",), "close {target}"),
    "examine": ("Inspect an accessible object or receptacle at your current location. Does not move you there.", ("target",), "examine {target}"),
    "take": ("Take a visible object from the receptacle at your current location. A closed container must be opened first. You can hold only one object at a time.", ("object", "source"), "take {object} from {source}"),
    "put": ("Place your held object in or on a receptacle at your current location. Open a closed destination first.", ("object", "destination"), "move {object} to {destination}"),
    "use": ("Use an accessible object, such as switching on a lamp at your current location. To examine an object under a lamp, hold the object when using the lamp.", ("target",), "use {target}"),
    "clean": ("Clean your held object at a sinkbasin. Move to the sinkbasin first.", ("object", "receptacle"), "clean {object} with {receptacle}"),
    "heat": ("Heat your held object using a microwave at your current location.", ("object", "receptacle"), "heat {object} with {receptacle}"),
    "cool": ("Cool your held object using a fridge at your current location.", ("object", "receptacle"), "cool {object} with {receptacle}"),
}
# Maps executor names to the names advertised in the tool schema.
SCHEMA_NAMES = {name: name + "_object" for name in ("clean", "heat", "cool")}


def tool_schemas():
    schemas = []
    for name, (description, fields, _) in REGISTRY.items():
        schemas.append({"type": "function", "function": {
            "name": SCHEMA_NAMES.get(name, name),
            "description": description,
            "parameters": {"type": "object", "properties": {
                field: {"type": "string", "description": "Exact identifier from observations, including its number."}
                for field in fields
            }, "required": list(fields), "additionalProperties": False},
        }})
    return copy.deepcopy(schemas)


class ToolCallError(ValueError):
    pass


def command_for(name, arguments):
    if name not in REGISTRY:
        raise ToolCallError("unknown_tool")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            raise ToolCallError("invalid_arguments: expected a JSON object") from None
    fields = REGISTRY[name][1]
    if not isinstance(arguments, dict) or set(arguments) != set(fields):
        raise ToolCallError("invalid_arguments: required fields are " + ", ".join(fields))
    for value in arguments.values():
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n\x00"):
            raise ToolCallError("invalid_arguments: identifiers must be nonempty single-line strings")
    return REGISTRY[name][2].format(**{key: value.strip() for key, value in arguments.items()})
