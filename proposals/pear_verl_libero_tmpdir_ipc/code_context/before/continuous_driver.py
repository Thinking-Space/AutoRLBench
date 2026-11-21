"""Inactive, admitted dedicated 2+6 continuous driver; no finite step target.

build_admitted_plan refuses a missing/unaccepted CP15. Importing this module
does not import Ray, create jobs, prepare tool config, or touch GPUs.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from continuous_loop import atomic_object, bootstrap_binding, continuous_horizon, owned, read_object


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "work/runs/clean-rl-smoke"
PHASE = "continuous"
POD = "node-launcher"
PREFIX = "eight-node-continuous"
INDEX = "evidence/libero-continuous/latest-accepted.json"
CONTROL = "work/runtime/libero-continuous/control.json"
TICKET = re.compile(r"[0-9a-f]{32}")


class NotReady(RuntimeError):
    """A real complete accepted engineering handoff does not exist yet."""


def module(relative, name):
    path = owned(ROOT, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _base():
    # The reused standalone smoke module imports this ROOT's clean_tool_config.
    sys.path.insert(0, str(ROOT))
    return module("work/dev/libero-hsdp-resume/eight_node_driver.py", "continuous_base_helpers")


def _legacy():
    return module("work/dev/libero-engineering10-verify/engineering10_verify.py", "continuous_legacy_acceptance")


def _auditor():
    return module("work/dev/libero-continuous-scheduler/continuous_step_audit.py", "continuous_resume_acceptance")


def paths(ticket):
    if TICKET.fullmatch(ticket) is None:
        raise ValueError("A continuous attempt needs an exact immutable ticket")
    stem = f"evidence/libero-continuous/{ticket}"
    return {name: stem + "/" + name + ".json" for name in
            ("launch", "config-check", "initial-publication", "binding", "stop")}


def _marker():
    path = owned(ROOT, RUN / "ckpt/latest_checkpointed_iteration.txt")
    with path.open("rb") as stream:
        raw = stream.read(33)
    if len(raw) > 32 or not raw.strip().isdigit():
        raise ValueError("The current continuous checkpoint marker is invalid")
    return int(raw)


def _bytes(value, limit=256 * 1024):
    path = owned(ROOT, value)
    if path.stat().st_size > limit:
        raise ValueError("Continuous metadata exceeds its bounded read size")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Continuous metadata grew beyond its read bound")
    return raw


def admit_resume():
    step = _marker()
    if step < 15:
        raise NotReady(f"Actual engineering checkpoint {step} is below the accepted CP15 milestone")
    if step == 15:
        controller, _ = read_object(ROOT, "evidence/eight-node-engineering10-controller.json")
        if (controller.get("status") != "exited" or controller.get("compute_returncode") != 0
                or controller.get("passed") is not True or controller.get("keepalive_restored") is not True):
            raise NotReady("CP15 is not yet at its accepted controller/guard terminal handoff")
        verify = _legacy()
        checkpoint = verify.verify_committed_restart(ROOT, 15)
        completion = verify.verify_completion(ROOT, controller)
        if completion.get("passed") is not True or completion.get("cumulative_updates") != 15:
            raise NotReady("CP15 has not passed actual publication and owned terminal handoff")
        reference = "evidence/eight-node-engineering10-step15-acceptance.json"
        source, digest = read_object(ROOT, reference)
        if source.get("passed") is not True or source.get("step") != 15 or source.get("checkpoint") != checkpoint:
            raise ValueError("Legacy CP15 handoff differs from its accepted actual checkpoint")
    else:
        index, _ = read_object(ROOT, INDEX)
        if index.get("last_accepted_step") != step:
            raise NotReady("The newest saved continuous checkpoint lacks exact publication acceptance")
        reference, digest = index["acceptance_path"], index["acceptance_sha256"]
        source = _auditor().verify_committed_restart(ROOT, reference, digest, expected_step=step)
        checkpoint = source["checkpoint"]
    if checkpoint["global_step"] != step or checkpoint["world_size"] != 16:
        raise ValueError("Continuous resume requires an accepted sixteen-rank checkpoint")
    if checkpoint["checkpoint_path"] != str(RUN / "ckpt" / f"global_step_{step}"):
        raise ValueError("Continuous resume checkpoint is not the exact current run")
    return checkpoint, {"path": reference, "sha256": digest}, source


def _validate_registry(smoke):
    registry = smoke.read_json(ROOT / "config/cluster-eight-h20-20261118.json")
    if (len(registry["hosts"]) != 8 or len({host["pod"] for host in registry["hosts"]}) != 8
            or [host.get("role") for host in registry["hosts"]].count("trainer") != 2
            or [host.get("role") for host in registry["hosts"]].count("rollout") != 6
            or any(len(host["gpus"]) != 8 for host in registry["hosts"])
            or len({gpu["uuid"] for host in registry["hosts"] for gpu in host["gpus"]}) != 64):
        raise ValueError("Continuous resource admission lost its exact eight Pod/64 UUID registry")


def build_admitted_plan(ticket, gate=None):
    paths(ticket)
    checkpoint, source, _ = admit_resume()  # Before tool/profile writes or any job creation.
    return _build_configuration_plan(ticket, checkpoint, source, gate=gate)


def _build_configuration_plan(ticket, checkpoint, source, *, gate=None):
    """Build only argv/config text; callers choose real admission or explicit CPU probe."""
    location = paths(ticket)
    base = _base()
    smoke = base.smoke
    _validate_registry(smoke)
    cluster = smoke.read_json(ROOT / "config/cluster-candidate-a-16env.json")
    cluster.update(trainer_nodes=2, rollout_nodes=6, ray_address="<node-address>:17679",
                   model_dir=str(ROOT / "work/models/Qwen3.8-27B"),
                   work_dir=str(ROOT / "work"), run_dir=str(RUN))
    environment, command = smoke.resolve("smoke", cluster)
    runtime_index = command.index("--runtime-dir") + 1
    command[runtime_index] = str(ROOT / "work/runtime" / POD / f"continuous-{ticket}-ipc")
    # A fresh profile avoids modifying a tool config belonging to the healthy current ticket.
    tool = smoke.prepare_tool_config(ROOT, "smoke-continuous-" + ticket[:16], 96)
    namespace = "pear-eight-h20-continuous-" + ticket[:16]
    metrics = RUN / f"metrics-continuous-from-{checkpoint['global_step']}-{ticket[:16]}.jsonl"
    audit = RUN / "actor_audit" / ticket
    helper = ROOT / "work/dev/libero-continuous-scheduler"
    pythonpath = os.pathsep.join([str(helper), environment.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))])
    environment.update(
        CUDA_VISIBLE_DEVICES="", NO_GPU="1", STEPS="null", PYTHONDONTWRITEBYTECODE="1",
        TRITON_BACKENDS_IN_TREE="1", LOGGER='["console","file","tensorboard","wandb"]',
        EPISODES=str(ROOT / "source/pear/examples/clean/stage1.json"), PYTHONPATH=pythonpath,
        PEAR_TQ_NAMESPACE=namespace, PEAR_SIM_POOL_NAMESPACE=namespace,
        VERL_FILE_LOGGER_PATH=str(metrics), PEAR_CONTINUOUS_TICKET=ticket,
        TENSORBOARD_DIR=str(ROOT / "reports/tensorboard/libero/continuous" / ticket),
        WANDB_MODE="offline", WANDB_DIR=str(ROOT / "reports/wandb/libero/continuous" / ticket),
        WANDB_CONFIG_DIR=str(ROOT / "work/cache/wandb-config"),
        WANDB_CACHE_DIR=str(ROOT / "work/cache/wandb"), WANDB_DATA_DIR=str(ROOT / "work/cache/wandb-data"),
        WANDB_DISABLE_GIT="true", WANDB_DISABLE_CODE="true", WANDB__DISABLE_STATS="true",
        WANDB__DISABLE_META="true", WANDB__DISABLE_MACHINE_INFO="true",
        TMPDIR=str(ROOT / "work/tmp"), TEMP=str(ROOT / "work/tmp"), TMP=str(ROOT / "work/tmp"),
    )
    continuous = {"enabled": True, "binding_path": str(ROOT / location["binding"]), "binding_sha256": None,
                  "control_path": str(ROOT / CONTROL), "state_path": str(ROOT / INDEX),
                  "milestone_dir": str(ROOT / "evidence/libero-continuous/milestones"), "milestone_steps": [15]}
    serialized = "{" + ",".join(key + ":" + json.dumps(value, separators=(",", ":"))
                                  for key, value in continuous.items()) + "}"
    overrides = [
        "data.custom_cls.path=" + str(ROOT / "source/pear/pear/agent_loop/dataset.py"),
        "trainer.total_training_steps=null", "trainer.resume_mode=resume_path",
        "trainer.resume_from_path=" + checkpoint["checkpoint_path"],
        "trainer.save_freq=1", "trainer.max_actor_ckpt_to_keep=1",
        "actor_rollout_ref.hybrid_engine=false", "trainer.idle_occupancy.enabled=true",
        "trainer.idle_occupancy.binary=" + str(ROOT.parent / "gpu-keepalive/h20-20261116/occupy"),
        "trainer.idle_occupancy.directory=" + str(RUN / "trainer-idle-continuous"),
        "trainer.idle_occupancy.activation_delay=120",
        "++trainer.checkpoint_storage={directory:" + str(ROOT.parent)
        + ",limit_bytes:4000000000000,first_checkpoint_reserve_bytes:400000000000}",
        "++trainer.continuous=" + serialized,
        "++actor_rollout_ref.actor.fsdp_config.update_audit_dir=" + str(audit),
        "+trainer.v1.sampler.require_latest_policy=true", "trainer.v1.sampler.max_off_policy_threshold=1",
        "trainer.v1.sampler.max_off_policy_strategy=drop", "trainer.v1.separate_async.parameter_sync_step=1",
        "actor_rollout_ref.actor.optim.lr=1e-6", "actor_rollout_ref.actor.optim.lr_scheduler_type=constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0", "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0",
        "actor_rollout_ref.actor.optim.total_training_steps=-1",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=2", "reward.num_workers=1",
        "++data.apply_chat_template_kwargs.enable_thinking=true",
        "++data.apply_chat_template_kwargs.reasoning_effort=xhigh",
        "actor_rollout_ref.rollout.multi_turn.tool_config_path=" + tool["path"],
    ]
    forwarded = ("TRITON_BACKENDS_IN_TREE", "VERL_FILE_LOGGER_PATH", "PEAR_TQ_NAMESPACE", "PEAR_SIM_POOL_NAMESPACE",
                 "TENSORBOARD_DIR", "WANDB_MODE", "WANDB_DIR", "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR",
                 "WANDB_DATA_DIR", "WANDB_DISABLE_GIT", "WANDB_DISABLE_CODE", "WANDB__DISABLE_STATS",
                 "WANDB__DISABLE_META", "WANDB__DISABLE_MACHINE_INFO", "PYTHONPATH", "PEAR_CONTINUOUS_TICKET",
                 "TMPDIR", "TEMP", "TMP")
    overrides += ["++ray_kwargs.ray_init.runtime_env.env_vars." + key + "=" + json.dumps(environment[key])
                  for key in forwarded]
    overrides.append("++ray_kwargs.ray_init.runtime_env.env_vars.WORK_DIR=" + str(ROOT / "work"))
    if gate is not None:
        overrides.append(smoke.startup_gate_override(json.dumps(gate)))
    return {"phase": PHASE, "layout_mode": "dedicated", "attempt_id": ticket, "ticket": ticket,
            "run_dir": str(RUN), "target_step": None, "target_checkpoint_step": None,
            "checkpoint": checkpoint, "checkpoint_state": {**checkpoint, "starting_step": checkpoint["global_step"]},
            "source_acceptance": source, "initial_publication_path": str(ROOT / location["initial-publication"]),
            "binding_path": str(ROOT / location["binding"]), "metrics_path": str(metrics), "audit_dir": str(audit),
            "environment": environment, "command": command + overrides, "tool": tool,
            "evidence_prefix": PREFIX + "-" + ticket, "paths": location}


def build_configuration_probe(ticket):
    """CPU schema probe on the actual current accepted checkpoint, with no admission.

    In particular a real CP5 remains CP5; this never fabricates a CP15 restore
    or publishes launch/binding documents. Only --cfg consumes this plan.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("NO_GPU") != "1":
        raise ValueError("Configuration probes require hidden GPUs")
    step = _marker()
    if not 5 <= step <= 15:
        raise NotReady("The initial configuration probe expects the known engineering 5–15 lineage")
    checkpoint = _legacy().verify_committed_restart(ROOT, step)
    reference = ("evidence/eight-node-engineering10-cp5-anchor.json" if step == 5 else
                 f"evidence/eight-node-engineering10-step{step}-acceptance.json")
    _, digest = read_object(ROOT, reference)
    plan = _build_configuration_plan(ticket, checkpoint, {"path": reference, "sha256": digest})
    plan.update(phase="configuration_probe", admission=False,
                evidence_prefix="eight-node-continuous-config-probe-" + ticket)
    return plan


def check_configuration(plan):
    """Real Hydra + RunConfig + FSDP dataclass, only when explicitly invoked."""
    from omegaconf import OmegaConf, open_dict
    from pear.trainer.config import RunConfig
    from verl.utils.config import omega_conf_to_dataclass
    log = ROOT / "evidence" / (plan["evidence_prefix"] + "-hydra.log")
    with log.open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        subprocess.run([*plan["command"], "--cfg", "job", "--resolve"], cwd=ROOT / "source/pear",
                       env={**os.environ, **plan["environment"]}, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=600)
    if log.stat().st_size > 256 * 1024:
        raise ValueError("Continuous Hydra output exceeds its bounded size")
    lines = log.read_text().splitlines()
    first = next(index for index, line in enumerate(lines) if line.startswith("model_engine:"))
    config = OmegaConf.create("\n".join(lines[first:]))
    RunConfig(config).prepare()
    previous = {key: os.environ.get(key) for key in ("VERL_ENGINE_DEVICE", "VERL_ENGINE_VENDOR")}
    os.environ.update(VERL_ENGINE_DEVICE="cuda", VERL_ENGINE_VENDOR="nvidia")
    try:
        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if getattr(actor.checkpoint, "expand_dp_replicas_from", None) is not None:
        raise ValueError("Continuous resume must use the same sixteen-rank checkpoint loader")
    continuous_horizon(config)
    expected = {
        "trainer.resume_from_path": plan["checkpoint"]["checkpoint_path"], "trainer.resume_mode": "resume_path",
        "trainer.del_local_ckpt_after_load": False, "trainer.v1.trainer_mode": "separate_async",
        "trainer.continuous.binding_path": plan["binding_path"], "trainer.continuous.binding_sha256": None,
        "trainer.v1.separate_async.parameter_sync_step": 1, "trainer.checkpoint_storage.limit_bytes": 4_000_000_000_000,
        "actor_rollout_ref.hybrid_engine": False, "actor_rollout_ref.actor.fsdp_config.fsdp_size": 8,
        "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": 8,
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.async_save": False,
        "actor_rollout_ref.actor.fsdp_config.update_audit_dir": plan["audit_dir"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 4, "actor_rollout_ref.rollout.agent.num_workers": 24,
        "actor_rollout_ref.rollout.topk_log_probs": 32, "trainer.v1.sampler.max_inflight_sessions": 96,
        "algorithm.rollout_correction.score_centering": True,
        "actor_rollout_ref.rollout.temperature": 1.0, "actor_rollout_ref.rollout.top_p": 1.0,
        "data.max_prompt_length": 6144, "data.max_response_length": 160000,
        "actor_rollout_ref.model.path": plan["environment"]["MODEL"],
        "data.custom_cls.path": str(ROOT / "source/pear/pear/agent_loop/dataset.py"),
    }
    for key, value in expected.items():
        if OmegaConf.select(config, key) != value:
            raise ValueError("Unexpected continuous effective configuration: " + key)
    if config.trainer.get("engineering10_attempt_id") or config.trainer.get("formal_restart_audit", False):
        raise ValueError("Continuous initial publication cannot inherit another phase's hook")
    _base().smoke.verify_tool_config(plan["tool"], config.actor_rollout_ref.rollout.multi_turn.tool_config_path)
    with open_dict(config.actor_rollout_ref.rollout.custom):
        config.actor_rollout_ref.rollout.custom.pop("startup_gate", None)
    effective = ROOT / "evidence" / (plan["evidence_prefix"] + "-effective.yaml")
    OmegaConf.save(config, effective, resolve=True)
    os.chmod(effective, 0o600)
    return {"passed": True, "config_sha256": hashlib.sha256(effective.read_bytes()).hexdigest(),
            "effective_path": str(effective), "starting_checkpoint": plan["checkpoint"],
            "checked_values": expected, "optimizer_updates": 0, "mode": plan["phase"],
            "admission": plan.get("admission", True)}


def publish_launch(plan, configuration):
    if plan.get("phase") != PHASE or plan.get("admission", True) is not True:
        raise ValueError("A CPU configuration probe cannot publish a production launch or binding")
    location = paths(plan["ticket"])
    raw = _bytes(configuration["effective_path"])
    if hashlib.sha256(raw).hexdigest() != configuration["config_sha256"]:
        raise ValueError("Continuous resolved configuration changed before launch publication")
    launch = {key: plan[key] for key in ("phase", "layout_mode", "attempt_id", "ticket", "run_dir",
              "target_step", "checkpoint", "source_acceptance", "initial_publication_path", "binding_path",
              "metrics_path", "audit_dir", "environment")}
    launch.update(started_at=time.time(), configuration=configuration, starting_checkpoint=plan["checkpoint"])
    atomic_object(ROOT, location["config-check"], configuration, exclusive=True)
    atomic_object(ROOT, location["launch"], launch, exclusive=True)
    atomic_object(ROOT, f"evidence/{PREFIX}-launch.json", launch)
    return launch


def publish_initial(trainer, synchronize):
    """Verify actual restore, block physical sync, freeze identities, then bootstrap the fit hook."""
    import ray
    from pear.trainer import formal_recovery
    settings = trainer.config.trainer.continuous
    binding_path = owned(ROOT, settings.binding_path)
    ticket = binding_path.parent.name
    location = paths(ticket)
    launch, launch_sha = read_object(ROOT, location["launch"])
    checkpoint = launch["starting_checkpoint"]
    step = trainer.global_steps
    if (launch["layout_mode"] != "dedicated" or launch["phase"] != PHASE
            or launch.get("ticket") != ticket or launch.get("attempt_id") != ticket
            or launch["run_dir"] != str(RUN)
            or launch["target_step"] is not None or step != checkpoint["global_step"] or _marker() != step
            or trainer.config.trainer.resume_mode != "resume_path"
            or trainer.config.trainer.resume_from_path != checkpoint["checkpoint_path"]
            or getattr(trainer, "_enable_hybrid_replicas", False)):
        raise ValueError("Continuous initial publication lost the exact dedicated accepted resume source")
    source, source_sha = read_object(ROOT, launch["source_acceptance"]["path"])
    if source_sha != launch["source_acceptance"]["sha256"]:
        raise ValueError("Continuous accepted source changed before actual restore")
    restored, sessions = {}, {}
    handoff = module("work/dev/libero-hsdp-handoff/hsdp_handoff.py", "continuous_initial_handoff")
    for rank in range(16):
        row = formal_recovery.last_actor_event(Path(launch["audit_dir"]) / f"rank-{rank}.jsonl")
        state = formal_recovery.restore_identity(row, rank, step, Path(checkpoint["checkpoint_path"]))
        expected = source["rank_references"][str(rank)]["saved_state"]
        if (state["parameter_sha256"] != expected["parameter_sha256"]
                or row.get("optimizer_steps") != expected["optimizer_steps"]
                or row.get("rng") != expected["rng"]
                or type(row.get("pid")) is not int or row["pid"] <= 1):
            raise ValueError(f"Continuous actual restored state differs on rank {rank}")
        handoff.check_counters(row["optimizer_steps"], step)
        restored[str(rank)] = {**state, "pid": row["pid"], "load_at": row["time"],
                               "optimizer_steps": row["optimizer_steps"], "rng_restored": True}
        sessions[str(rank)] = {"pid": row["pid"], "session_id": state["session_id"]}
    if len({row["session_id"] for row in sessions.values()}) != 16:
        raise ValueError("Continuous publication requires sixteen distinct restored actor sessions")
    claim, _ = read_object(ROOT, f"evidence/formal-gcs/{ticket}-driver.json")
    job = ray.get_runtime_context().get_job_id()
    if claim["ticket"] != ticket or claim["job_id"] != job:
        raise ValueError("Continuous publication belongs to another Ray driver")
    handles = trainer.standalone_server_manager.server_handles
    if len(handles) != 12:
        raise ValueError("Continuous dedicated initial publication needs twelve fixed leaders")
    started_at, timer = time.time(), time.perf_counter()
    if any(not claim["at"] <= row["load_at"] <= started_at for row in restored.values()):
        raise ValueError("A continuous restored actor event is outside this exact training job")
    synchronize()
    seconds = time.perf_counter() - timer
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Continuous initial synchronization did not return positive elapsed time")
    versions = formal_recovery.check_installed_versions(
        ray.get([handle.get_installed_policy_version.remote() for handle in handles], timeout=120), step)
    publication = {"schema_version": 1, "passed": True, "phase": PHASE, "ticket": ticket, "attempt_id": ticket,
                   "job_id": job, "run_dir": launch["run_dir"], "step": step, "published_policy": step,
                   "next_training_step": step + 1, "checkpoint_path": checkpoint["checkpoint_path"],
                   "restored_ranks": restored, "replica_versions": versions,
                   "launch_sha256": launch_sha, "launch_config_sha256": launch["configuration"]["config_sha256"],
                   "started_at": started_at, "finished_at": time.time(), "weight_sync_seconds": seconds}
    atomic_object(ROOT, location["initial-publication"], publication, exclusive=True)
    _, publication_sha = read_object(ROOT, location["initial-publication"])
    binding = {"schema_version": 1, "ticket": ticket, "job_id": job,
               "run_dir": str(RUN.relative_to(ROOT)), "metrics_path": str(Path(launch["metrics_path"]).relative_to(ROOT)),
               "audit_dir": str(Path(launch["audit_dir"]).relative_to(ROOT)), "source_step": step,
               "source_acceptance": launch["source_acceptance"], "config_sha256": publication["launch_config_sha256"],
               "initial_publication": {"path": location["initial-publication"], "sha256": publication_sha},
               "actor_sessions": sessions, "layout_mode": "dedicated",
               "rollout_leaders": [{key: row[key] for key in ("actor_id", "replica_rank", "node_rank")} for row in versions],
               "rollout_layout": {"dedicated_nodes": 6, "gpus_per_node": 8, "tensor_parallel_size": 4,
                                  "hybrid_gpu_count": 0}}
    atomic_object(ROOT, location["binding"], binding, exclusive=True)
    _, binding_sha = read_object(ROOT, location["binding"])
    bootstrap_binding(trainer, binding_sha)
    return publication


def verify_completed_run(run, phase):
    if Path(run) != RUN or phase != PHASE:
        raise ValueError("Continuous stop completion must bind the exact engineering run")
    controller, _ = read_object(ROOT, "evidence/eight-node-continuous-controller.json")
    ticket = controller["ray_gcs"]["ticket"]
    stop, stop_sha = read_object(ROOT, paths(ticket)["stop"])
    control, digest = read_object(ROOT, stop["control_path"])
    if (controller.get("status") != "exited" or controller.get("compute_returncode") != 0
            or stop.get("passed") is not True or stop.get("ticket") != ticket
            or stop.get("kind") != "safe continuous stop" or digest != stop["control_sha256"]
            or control.get("action") != "stop" or control.get("request_id") != stop["request_id"]
            or control.get("generation") != stop["generation"] or _marker() != stop["step"]):
        raise ValueError("Continuous exit is not a completed exact explicit user stop")
    if stop["step"] == 15:
        source, source_sha = read_object(ROOT, stop["acceptance_path"])
        checkpoint = _legacy().verify_committed_restart(ROOT, 15)
        if source_sha != stop["acceptance_sha256"] or source.get("checkpoint") != checkpoint:
            raise ValueError("Pre-update continuous stop differs from its initial accepted CP15")
    else:
        source = _auditor().verify_committed_restart(ROOT, stop["acceptance_path"], stop["acceptance_sha256"],
                                                    expected_step=stop["step"])
        checkpoint = source["checkpoint"]
    handoff = module("work/dev/libero-hsdp-handoff/hsdp_handoff.py", "continuous_stop_handoff")
    direct = module("work/dev/libero-formal-restart/formal_gcs.py", "continuous_stop_gcs")
    direct.check_export_release(ROOT, controller)
    registry = handoff.read(ROOT, handoff.REGISTRY)
    expected = {host["pod"]: {gpu["uuid"] for gpu in host["gpus"]} for host in registry["hosts"]}
    restoration = controller["restoration"]
    if (set(restoration) != set(expected) or any(row.get("restored") is not True
            or type(row.get("guard_pid")) is not int or row["guard_pid"] <= 1
            or len(row.get("gpus", [])) != 8
            or {gpu[1] for gpu in row.get("gpus", [])} != expected[pod]
            or any(float(gpu[2]) < 30 or not handoff.positive(float(gpu[2])) for gpu in row.get("gpus", []))
            for pod, row in restoration.items())):
        raise ValueError("Continuous safe stop requires all eight exact guards actually active")
    return {"passed": True, "kind": "explicit continuous stop", "cumulative_updates": stop["step"],
            "checkpoint": checkpoint, "stop_receipt": paths(ticket)["stop"], "stop_sha256": stop_sha,
            "terminal_gcs_proof": controller["gcs_terminal"], "guards_restored": 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("NO_GPU") != "1":
        parser.error("Continuous admission is CPU only")
    gate = json.loads(os.environ["PEAR_GPU_STARTUP_GATE"]) if args.execute else None
    if args.execute and (os.environ.get("POD_NAME") != POD
                         or os.environ.get("PEAR_CLEAN_GUARDED_STAGE") != PHASE
                         or os.environ.get("PEAR_FORMAL_GCS_TICKET") != args.attempt_id):
        parser.error("Execution requires this launcher's exact owned continuous controller")
    plan = build_admitted_plan(args.attempt_id, gate=gate)
    if not args.execute:
        print(json.dumps({key: plan[key] for key in ("phase", "layout_mode", "ticket", "target_step", "checkpoint",
                                                     "source_acceptance", "metrics_path", "binding_path")}))
        return
    os.umask(0o077)
    lock = (RUN / ".smoke-driver.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    configuration = check_configuration(plan)
    if admit_resume()[0] != plan["checkpoint"]:
        raise ValueError("The accepted checkpoint changed during continuous CPU configuration validation")
    if owned(ROOT, plan["metrics_path"]).exists():
        raise ValueError("This exact continuous attempt already has a metrics segment")
    publish_launch(plan, configuration)
    os.set_inheritable(lock.fileno(), True)
    os.chdir(ROOT / "source/pear")
    os.execvpe(plan["command"][0], plan["command"], {**os.environ, **plan["environment"]})


if __name__ == "__main__":
    main()
