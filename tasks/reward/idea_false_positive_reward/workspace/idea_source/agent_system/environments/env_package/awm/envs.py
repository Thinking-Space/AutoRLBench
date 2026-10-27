import json
import os
import subprocess
import signal
from pathlib import Path
from typing import List


class AWMMCPVectorEnv:
    """Vector environment backed by isolated AgentWorldModel MCP servers."""

    def __init__(self, env_num: int, group_n: int, env_config) -> None:
        self.batch_size = env_num * group_n
        self.config = env_config
        self._process = None
        self._log = None
        self._start_bridge()

    def _start_bridge(self) -> None:
        awm_root = Path(str(self.config.awm.root)).expanduser().resolve()
        # Do not resolve this symlink: Python uses the venv entry path to locate site-packages.
        python = Path(str(self.config.awm.python)).expanduser().absolute()
        bridge = Path(__file__).with_name("bridge.py").resolve()
        if not python.is_file():
            raise FileNotFoundError(f"AWM Python not found: {python}")
        log_dir = Path(str(self.config.awm.work_root)).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = (log_dir / f"bridge_{os.getpid()}_{id(self)}.log").open("a", encoding="utf-8")
        # Nested conda/Ray state breaks mcp-agent's configuration bootstrap.
        # Keep the bridge isolated from the parent conda/Ray setup, but retain
        # proxy and CA settings needed to reach hosted judge endpoints.
        inherited_keys = (
            "HOME", "USER", "LANG", "TERM", "SHELL", "TMPDIR", "TMP", "TEMP",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
        )
        env = {key: os.environ[key] for key in inherited_keys if key in os.environ}
        env["PATH"] = f"{python.parent}:/usr/local/bin:/usr/bin:/bin"
        env["PYTHONPATH"] = str(awm_root)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        local_no_proxy = "localhost,127.0.0.1,::1"
        for key in ("NO_PROXY", "no_proxy"):
            existing = env.get(key, "")
            env[key] = f"{existing},{local_no_proxy}" if existing else local_no_proxy
        self._process = subprocess.Popen(
            [str(python), str(bridge)], cwd=str(awm_root), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1, start_new_session=True,
        )
        config = {
            "awm_root": str(awm_root),
            "data_root": str(Path(str(self.config.awm.data_root)).expanduser().resolve()),
            "work_root": str(log_dir / f"sessions_{os.getpid()}_{id(self)}"),
            "max_steps": int(self.config.max_steps),
            "tool_timeout": float(self.config.awm.tool_timeout),
            "reward_mode": str(self.config.awm.get("reward_mode", "code")),
            "judge_base_url": str(self.config.awm.get("judge_base_url", "")),
            "judge_api_key": str(self.config.awm.get("judge_api_key", "EMPTY")),
            "judge_model": str(self.config.awm.get("judge_model", "")),
            "judge_concurrency": int(self.config.awm.get("judge_concurrency", 10)),
            "judge_timeout": float(self.config.awm.get("judge_timeout", 120)),
            "judge_max_retries": int(self.config.awm.get("judge_max_retries", 3)),
            "startup_concurrency": int(self.config.awm.get("startup_concurrency", 32)),
            "repeat_success_terminate_k": int(self.config.awm.get("repeat_success_terminate_k", 5)),
        }
        self._process.stdin.write(json.dumps(config) + "\n")
        self._process.stdin.flush()

    def _request(self, payload: dict) -> dict:
        if self._process is None or self._process.poll() is not None:
            code = None if self._process is None else self._process.returncode
            raise RuntimeError(f"AWM MCP bridge is not running (exit code {code})")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("AWM MCP bridge closed its output unexpectedly")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "AWM MCP bridge request failed"))
        return response

    def reset(self, kwargs: List[dict]):
        if len(kwargs) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} reset items, got {len(kwargs)}")
        items = []
        for item in kwargs:
            if hasattr(item, "item") and not isinstance(item, dict):
                item = item.item()
            items.append(dict(item))
        response = self._request({"command": "reset", "items": items})
        return response["tasks"], [{"scenario": item["scenario"]} for item in items]

    def step(self, actions: List[dict]):
        response = self._request({"command": "step", "actions": actions})
        results = response["results"]
        observations = [result["observation"] for result in results]
        rewards = [float(result["reward"]) for result in results]
        dones = [bool(result["done"]) for result in results]
        infos = []
        for result, action in zip(results, actions):
            infos.append({
                "won": bool(result["reward"] >= 1.0),
                "tool_calling": int(action.get("name") in {"list_tools", "call_tool"}),
                "reward_type": result.get("reward_type", ""),
                "verify_result": result.get("verify_result", {}),
                "is_action_valid": bool(result.get("action_valid", True)),
            })
        return observations, rewards, dones, infos

    def close(self):
        if self._process is None:
            return
        if self._process.poll() is None:
            try:
                self._request({"command": "close"})
                self._process.wait(timeout=10)
            except Exception:
                self._terminate_process_group()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(force=True)
                    self._process.wait()
        for pipe in (self._process.stdin, self._process.stdout):
            if pipe:
                pipe.close()
        if self._log:
            self._log.close()
        self._process = None

    def _terminate_process_group(self, force: bool = False):
        if self._process is None or self._process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            (self._process.kill if force else self._process.terminate)()

    def close_sessions(self):
        if self._process is None or self._process.poll() is not None:
            return
        try:
            self._request({"command": "close_sessions"})
        except Exception:
            self._terminate_process_group()

    def __del__(self):
        self.close()


def build_awm_envs(env_num: int, group_n: int, env_config, **_) -> AWMMCPVectorEnv:
    return AWMMCPVectorEnv(env_num=env_num, group_n=group_n, env_config=env_config)
