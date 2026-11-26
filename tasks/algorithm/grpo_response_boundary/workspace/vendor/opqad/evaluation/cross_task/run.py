"""One independent GPU shard; continuous batching and durable per-sample output."""

import fcntl, hashlib, json, os, shutil, sys, time
from argparse import Namespace
from pathlib import Path

S = Path(__file__).resolve().parent
H = Path(os.environ["OPQAD_EVAL_WORKDIR"]).expanduser().resolve()


def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def main():
    arm, shard = sys.argv[1], int(sys.argv[2])
    num_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    assert num_shards in (2, 4) and 0 <= shard < num_shards
    manifest = json.loads((H / "manifest.json").read_text())
    a = manifest["arms"][arm]
    p = manifest["protocol"]
    out = H / "outputs" / arm / f"shard{shard}"
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "RUN.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = dict(arm=a, protocol=p, shard=shard)
    if num_shards != 2:
        protocol["num_shards"] = num_shards
    if (out / "protocol.json").exists():
        assert json.loads((out / "protocol.json").read_text()) == protocol, "Protocol mismatch"
    atomic(out / "protocol.json", protocol)
    if (out / "GENERATION_DONE").exists():
        return
    model = Path(a["model"])
    assert (model / "config.json").exists() and list(model.glob("*.safetensors"))
    if a["method"] == "oscar_int2":
        os.environ.update(
            QAT_OSCAR_FULL_CUDAGRAPH="1",
            VLLM_OSCAR_QUANT_ALL_LAYERS="1",
            VLLM_OSCAR_PER_REQUEST_STAGING="1",
            VLLM_OSCAR_STAGE_MAX_SEQS=str(a["max_num_seqs"]),
            VLLM_OSCAR_GQA_DECODE="1",
            VLLM_OSCAR_GQA_VERIFY="1",
            VLLM_OSCAR_BF16_TAIL_MODE="2",
            VLLM_OSCAR_K_CLIP_RATIO="0.96",
            VLLM_OSCAR_V_CLIP_RATIO="0.92",
            VLLM_OSCAR_SINK_TOKENS="64",
            VLLM_OSCAR_RECENT_TOKENS="256",
            VLLM_OSCAR_K_ROTATION_PATH=a["rotation"] + "/k_rotation_qqt_r_h_pbr.pt",
            VLLM_OSCAR_V_ROTATION_PATH=a["rotation"] + "/v_rotation_sst_r_h_pbr.pt",
        )
    os.environ["VLLM_OSCAR_EXTRA_SKIP_LAYERS"] = ""
    os.environ["QAT_OSCAR_IDENTITY_ROTATION"] = "0"
    from transformers import AutoTokenizer
    from opqad.evaluation.eval_vllm_turboquant import _make_llm, _engine_config, _oscar_record
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    work = {}
    context = 0
    for b, setting in p["benches"].items():
        raw = (H / f"data/{b}.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == p["data_hashes"][b]
        rows = json.loads(raw)
        work[b] = []
        for i, r in enumerate(rows):
            # Tokenize every prompt before selecting shard so both engines use identical context limits.
            prompt = tok.apply_chat_template(
                [dict(role="user", content=r["prompt"])],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            # BroRL opens <think> in its template; Qwen3 opens it during generation.
            assert prompt.rstrip().endswith(("<think>", "<|im_start|>assistant")), (
                "Unexpected assistant prefix",
                arm,
                b,
                r["id"],
            )
            ids = tok.encode(prompt, add_special_tokens=False)
            context = max(context, len(ids) + setting["max_output_tokens"])
            if i % num_shards == shard:
                for sample in range(setting["n"]):
                    work[b].append((r, i, sample, ids))
    # No prompt truncation; reserve the full agreed output budget after each prompt.
    if context > 33792:
        raise ValueError("Prompt plus output budget exceeds the paper context limit")
    context = 33792
    args = Namespace(
        model=str(model),
        kv_cache_dtype=a["method"],
        tensor_parallel_size=1,
        max_model_len=context,
        max_num_seqs=a["max_num_seqs"],
        max_num_batched_tokens=65536,
        gpu_memory_utilization=0.90,
        seed=p["seed"],
        calculate_kv_scales=False,
        disable_q_quantization=False,
        enforce_eager=False,
    )
    llm = _make_llm(args)
    cfg = _engine_config(llm)
    skip = {int(x) for x in (cfg.cache_config.kv_cache_dtype_skip_layers or [])}
    layers = cfg.model_config.hf_text_config.num_hidden_layers
    assert str(cfg.cache_config.cache_dtype) == a["method"]
    assert not cfg.model_config.enforce_eager, "CUDA Graph disabled"
    graph_mode = str(cfg.compilation_config.cudagraph_mode)
    assert not graph_mode.endswith("NONE"), ("CUDA Graph mode disabled", graph_mode)
    expected_skip = {0, 1, layers - 2, layers - 1} if a["method"] == "turboquant_k2v3_nc" else set()
    assert skip == expected_skip, ("Wrong boundary layers", skip)
    engine = dict(
        model=str(model),
        kv=a["method"],
        skip_layers=sorted(skip),
        context=context,
        max_num_seqs=a["max_num_seqs"],
        cuda_graph=not cfg.model_config.enforce_eager,
        cudagraph_mode=graph_mode,
        compilation=str(cfg.compilation_config),
        oscar=_oscar_record(args, llm),
        job=os.environ.get("SLURM_JOB_ID"),
    )
    atomic(out / "engine.json", engine)
    print("VERIFIED_ENGINE", json.dumps(engine, default=str), flush=True)
    local = out / "journal"
    local.mkdir(exist_ok=True)
    try:
        # Short tasks finish first. Requests remain in the scheduler continuously, without batch barriers.
        for b in p["benches"]:
            target = out / f"{b}.jsonl"
            journal = local / f"{b}.jsonl"
            done = {}
            if target.exists():
                for line in target.open():
                    r = json.loads(line)
                    key = (r["id"], r["sample"])
                    assert key not in done
                    done[key] = r
                shutil.copyfile(target, journal)
            else:
                journal.write_text("")
            expected = {(r["id"], s) for r, _, s, _ in work[b]}
            assert set(done) <= expected
            requests = {}
            setting = p["benches"][b]
            for r, i, s, ids in work[b]:
                if (r["id"], s) in done:
                    continue
                params = SamplingParams(
                    n=1,
                    temperature=setting["temperature"],
                    top_p=setting["top_p"],
                    top_k=setting["top_k"],
                    max_tokens=setting["max_output_tokens"],
                    seed=p["seed"] + i * 100 + s,
                    stop_token_ids=[151643, 151645],
                    output_kind=RequestOutputKind.FINAL_ONLY,
                )
                request_id = f"{b}:{i}:{s}"
                llm.llm_engine.add_request(request_id, dict(prompt_token_ids=ids), params)
                requests[request_id] = (r, i, s, ids)
            start = time.time()
            last = start
            generated = 0
            dirty = False
            completed = len(done)
            print("START", arm, shard, b, len(requests), "requests", flush=True)
            with journal.open("a") as jf:
                while llm.llm_engine.has_unfinished_requests():
                    for rr in llm.llm_engine.step():
                        if not rr.finished:
                            continue
                        r, i, s, ids = requests.pop(rr.request_id)
                        assert len(rr.outputs) == 1
                        c = rr.outputs[0]
                        rec = dict(
                            id=r["id"],
                            sample=s,
                            response=c.text,
                            token_ids=list(c.token_ids),
                            finish_reason=c.finish_reason,
                            prompt_tokens=len(ids),
                            prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                            seed=p["seed"] + i * 100 + s,
                        )
                        line = json.dumps(rec, ensure_ascii=False) + "\n"
                        jf.write(line)
                        jf.flush()
                        os.fsync(jf.fileno())
                        if not dirty:
                            try:
                                with target.open("a") as fp:
                                    fp.write(line)
                                    fp.flush()
                                    os.fsync(fp.fileno())
                            except OSError as e:
                                dirty = True
                                print("SHARED_WRITE_FAILED_LOCAL_COPY_PRESERVED", journal, repr(e), flush=True)
                        generated += len(c.token_ids)
                        completed += 1
                        print(
                            "PROGRESS",
                            arm,
                            shard,
                            b,
                            completed,
                            "/",
                            len(expected),
                            "elapsed",
                            round(time.time() - start),
                            "tokens_finished",
                            generated,
                            flush=True,
                        )
                    if time.time() - last > 60:
                        print(
                            "HEARTBEAT",
                            arm,
                            shard,
                            b,
                            "pending",
                            len(requests),
                            "elapsed",
                            round(time.time() - start),
                            flush=True,
                        )
                        last = time.time()
            assert not requests and completed == len(expected)
            # Atomic final sync also heals a shared append failure; never mark DONE before it succeeds.
            temp = target.with_suffix(".sync")
            shutil.copyfile(journal, temp)
            temp.replace(target)
            atomic(
                out / f"{b}.generation.json",
                dict(
                    samples=completed,
                    new_tokens=generated,
                    elapsed_seconds=time.time() - start,
                    local_backup=str(journal),
                ),
            )
        (out / "GENERATION_DONE").write_text("All configured benchmarks complete\n")
    finally:
        print("ENGINE_SHUTDOWN_BEGIN", flush=True)
        llm.llm_engine.engine_core.shutdown(timeout=60)
        print("ENGINE_SHUTDOWN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
