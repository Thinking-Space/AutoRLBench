import logging
import os

import ray
import yaml

from miles.utils.tracking_utils.prometheus_utils import _COLLECTOR_ACTOR_NAME

logger = logging.getLogger(__name__)


def _resolve_ray_temp_dir() -> str:
    """Return this node's Ray temp dir (the ``--temp-dir`` root).

    Ray's temp dir is NOT always the default ``/tmp/ray``: launch scripts
    (``scripts/setup/ray_start.sh`` / ``env.sh``) set ``RAY_TMPDIR`` to e.g.
    ``/data/logs/tmp/ray/rank_0``. Both paths we care about are rooted here:

    * the Prometheus config Ray writes / we overwrite:
      ``<temp_dir>/session_latest/metrics/prometheus/prometheus.yml``
    * Ray's file_sd discovery json:
      ``<temp_dir>/prom_metrics_service_discovery.json`` (directly under the temp
      dir, NOT under ``session_latest/metrics`` — see Ray's ``metrics_head.py``:
      ``os.path.join(self.temp_dir, PROMETHEUS_SERVICE_DISCOVERY_FILE)``).

    Resolving this at runtime via Ray keeps the file we write in sync with the
    file the external Prometheus reads. Falls back to ``/tmp/ray`` on error.
    """
    try:
        rc = ray.get_runtime_context()
        temp_dir = rc.get_temp_dir()  # e.g. /tmp/ray or /data/logs/tmp/ray/rank_0
        if temp_dir:
            return temp_dir
    except Exception:
        pass
    return "/tmp/ray"


def _get_push_collector_target() -> str | None:
    """Return ``host:port`` of the ``--use-prometheus`` push collector's HTTP server.

    Looked up by name via ``ray.get_actor`` rather than assumed from the caller's
    own node, since ``update_prometheus_config`` may run on a different node than
    the (hard-pinned) collector actor. Returns ``None`` if ``--use-prometheus``
    was never passed (actor doesn't exist) or the actor isn't reachable.
    """
    try:
        collector = ray.get_actor(_COLLECTOR_ACTOR_NAME)
        return ray.get(collector.get_address.remote())
    except Exception:
        return None


def update_prometheus_config(args, server_addresses: list[str], rollout_name: str | None = None):
    """
    Update Prometheus configuration file with server addresses and reload on first node.

    Writes a ``prometheus.yml`` (scraping Ray's file_sd + the rollout servers' ``/metrics``,
    plus the ``--use-prometheus`` push collector's own HTTP server if one is running) to every
    alive Ray node, then POSTs ``/-/reload`` to the external Prometheus server on each node so
    it picks up the new scrape targets.

    The written config file lives under each node's ``<ray_tmpdir>/session_latest/metrics``
    (NOT always ``/tmp/ray`` — Ray's temp dir is configurable via ``RAY_TMPDIR``). The path is
    resolved per-node at write time via ``_resolve_ray_temp_dir()``
    (``ray.get_runtime_context().get_temp_dir()``), NOT from any CLI argument, so it always
    matches the file the external Prometheus launched by ``scripts/setup/setup_prometheus.sh``
    reads on that same node.

    Args:
        args: parsed miles args. Reads only ``prometheus_scrape_port`` (the external
            Prometheus server's ``/-/reload`` port).
        server_addresses: vllm or sglang server addresses, e.g. ``["host:port", ...]``.
        rollout_name: name of the rollout backend (e.g., "sglang", "vllm"). Used for the
            scrape job name.
    """

    if not server_addresses:
        logger.warning("No server addresses available to update Prometheus config")
        return

    port = args.prometheus_scrape_port
    push_collector_target = _get_push_collector_target()

    try:
        # Resolve each node's real Ray temp dir at write/reload time (inside the
        # per-node task), because RAY_TMPDIR differs per node. Both the config file
        # and the file_sd discovery json Ray writes are rooted there (at different
        # depths), so the scrape path is node-local.
        @ray.remote(num_cpus=0)
        def write_config_file():
            temp_dir = _resolve_ray_temp_dir()
            # Prometheus config lives under the session metrics dir; Ray's file_sd
            # discovery json lives directly under the temp dir (not the metrics dir).
            config_file = os.path.join(temp_dir, "session_latest", "metrics", "prometheus", "prometheus.yml")
            discovery_file = os.path.join(temp_dir, "prom_metrics_service_discovery.json")

            scrape_configs = [
                {
                    "job_name": "ray",
                    "file_sd_configs": [{"files": [discovery_file]}],
                },
                {"job_name": "rollout", "static_configs": [{"targets": server_addresses}]},
            ]
            if push_collector_target:
                # The --use-prometheus push collector (miles_metric_* gauges, e.g.
                # gateway/segments_total) runs its own standalone HTTP server that
                # is otherwise never scraped by anything.
                scrape_configs.append(
                    {
                        "job_name": "miles_push_collector",
                        "static_configs": [{"targets": [push_collector_target]}],
                    }
                )

            prometheus_config_json = {
                "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},
                "scrape_configs": scrape_configs,
            }

            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, "w") as f:
                yaml.dump(prometheus_config_json, f, default_flow_style=False, indent=2, sort_keys=False)
            return config_file

        # Reload Prometheus on all nodes. Only master node should succeed, skip errors on other nodes.
        @ray.remote(num_cpus=0)
        def reload_prometheus(port):
            import socket
            import subprocess

            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            reload_url = f"http://{ip_address}:{port}/-/reload"

            try:
                subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)
                print(f"Reloading Prometheus on node: {reload_url}")
            except Exception:
                # Skip errors on non-master nodes
                pass

        # Get all available nodes and schedule tasks on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]

        # Write config files on all nodes
        write_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = write_config_file.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote()
            write_tasks.append(task)

        written_files = ray.get(write_tasks)

        server_type = rollout_name.upper() if rollout_name else "rollout"
        print(
            f"Updated Prometheus configuration on {len(written_files)} node(s) with "
            f"{len(server_addresses)} {server_type} servers"
        )

        # Reload Prometheus on all nodes
        reload_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = reload_prometheus.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(port)
            reload_tasks.append(task)

        ray.get(reload_tasks)

    except Exception as e:
        logger.error(f"Failed to update Prometheus configuration: {e}")
