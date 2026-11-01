# -*- coding: utf-8 -*-
"""
Validation hooks for ray_trainer / reward_loop.

Installed from main_ppo_v0: wrap _validate with baseline
file paste, and ensure RewardLoopWorker.reinit_reward_fn exists.
"""

from __future__ import annotations

_INSTALLED = False


def install_bench_hooks() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_validate_paste()
    _install_reward_reinit()
    _INSTALLED = True


def _install_validate_paste() -> None:
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    except Exception as e:
        print(f"[bench] hooks: RayPPOTrainer unavailable: {e}", flush=True)
        return

    # Avoid double-wrap
    if getattr(RayPPOTrainer._validate, "_bench_wrapped", False):
        return

    original = RayPPOTrainer._validate

    def _validate(self, merged: bool = False):
        state = None
        try:
            from verl.utils.bench_baseline_swap import begin_baseline_validate

            state = begin_baseline_validate(
                reward_loop_manager=getattr(self, "reward_loop_manager", None)
            )
        except Exception as e:
            print(f"[bench] WARN: baseline validate paste failed: {e}", flush=True)
        try:
            return original(self, merged=merged)
        finally:
            try:
                from verl.utils.bench_baseline_swap import end_baseline_validate

                end_baseline_validate(state)
            except Exception as e:
                print(f"[bench] WARN: baseline validate restore failed: {e}", flush=True)

    _validate._bench_wrapped = True  # type: ignore[attr-defined]
    RayPPOTrainer._validate = _validate  # type: ignore[method-assign]
    print("[bench] hooks: wrapped RayPPOTrainer._validate for baseline paste", flush=True)


def _install_reward_reinit() -> None:
    try:
        from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
    except Exception as e:
        print(f"[bench] hooks: RewardLoopWorker unavailable: {e}", flush=True)
        return

    if getattr(RewardLoopWorker, "_bench_reinit_installed", False):
        return

    def reinit_reward_fn(self, swapped_rels=None):
        try:
            from verl.utils.bench_baseline_swap import purge_modules_for_rels

            if swapped_rels:
                purge_modules_for_rels(list(swapped_rels))
        except Exception as e:
            print(f"[bench] purge_modules failed: {e}", flush=True)

        try:
            import importlib

            import verl.trainer.ppo.reward as reward_mod

            importlib.reload(reward_mod)
            load_fn = reward_mod.load_reward_manager
        except Exception:
            from verl.trainer.ppo.reward import load_reward_manager as load_fn

        self.reward_manager = load_fn(
            self.config,
            self.input_tokenizer,
            reward_router_address=self.reward_router_address,
            reward_model_tokenizer=self.reward_model_tokenizer,
        )
        return {"ok": True, "swapped_rels": list(swapped_rels or [])}

    RewardLoopWorker.reinit_reward_fn = reinit_reward_fn  # type: ignore[method-assign]
    RewardLoopWorker._bench_reinit_installed = True  # type: ignore[attr-defined]
    print("[bench] hooks: installed RewardLoopWorker.reinit_reward_fn", flush=True)
