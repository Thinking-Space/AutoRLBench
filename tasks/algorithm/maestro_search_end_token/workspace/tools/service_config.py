"""Shared service settings exported by scripts/model_services.sh.

Defaults also allow tools to be imported independently of the shell launcher.
Only the five models offered in the prompt and the DeepEyes tool are deployed.
"""
from dataclasses import dataclass
import os


GPU_MEMORY_LIMIT = 0.94

DEFAULTS = {
    'qwen_vl': ('QWEN_VL', 'qwen3-VL-8B-Instruct', '0', 2362, 0.47, 18000, 16),
    'deepeyes': ('DEEPEYES', 'DeepEyes-7B', '0', 2370, 0.40, 18000, 8),
    'intern': ('INTERN', 'Intern-S1-mini', '1', 2368, 0.94, 18000, 32),
    'glm': ('GLM', 'GLM-4.6V-Flash', '2', 2376, 0.94, 18000, 32),
    'chart': ('CHART', 'Chart-R1', '3', 2364, 0.47, 8192, 16),
    'medgemma': ('MEDGEMMA', 'medgemma-1.5-4b-it', '3', 2369, 0.47, 8192, 16),
}


@dataclass(frozen=True)
class ModelService:
    key: str
    name: str
    model_dir: str
    served_model_name: str
    gpu: str
    port: int
    base_url: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    enforce_eager: bool


def get_service(key):
    prefix, name, gpu, port, memory, context, sequences = DEFAULTS[key]
    env = lambda suffix, default: os.environ.get(prefix + '_' + suffix, str(default))
    model_dir = env('MODEL_DIR', 'models/' + name)
    port = int(env('PORT', port))
    host = os.environ.get('MODEL_API_HOST', '127.0.0.1')
    return ModelService(
        key=key, name=name, model_dir=model_dir,
        served_model_name=env('SERVED_MODEL_NAME', 'models/' + name),
        gpu=env('GPU', gpu), port=port,
        base_url=env('BASE_URL', f'http://{host}:{port}/v1').rstrip('/'),
        gpu_memory_utilization=float(env('GPU_MEMORY_UTILIZATION', memory)),
        max_model_len=int(env('MAX_MODEL_LEN', context)),
        max_num_seqs=int(env('MAX_NUM_SEQS', sequences)),
        enforce_eager=key in {'qwen_vl', 'deepeyes', 'chart', 'medgemma'},
    )


def get_services():
    return [get_service(key) for key in DEFAULTS]


def base_url_for_port(port):
    for service in get_services():
        if service.port == int(port):
            return service.base_url
    raise ValueError(f'No deployed model service configured on port {port}')
