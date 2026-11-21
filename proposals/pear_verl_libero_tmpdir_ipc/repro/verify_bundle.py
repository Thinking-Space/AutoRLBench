"""Offline review-package integrity and evidence checks; never imports production code."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
def read(rel):return (ROOT/rel).read_text(encoding="utf-8")
def load(rel):return json.loads(read(rel))
def digest(data):return hashlib.sha256(data).hexdigest()
def forwarded(source):
    tree=ast.parse(source)
    matches=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="forwarded" for t in node.targets):
            matches.append(ast.literal_eval(node.value))
    assert len(matches)==1
    return matches[0]
def apply_patch(original,patch):
    lines=original.splitlines(True);patchlines=patch.splitlines(True)
    output=[];position=0;i=2
    assert patchlines[0].startswith("--- ") and patchlines[1].startswith("+++ ")
    while i<len(patchlines):
        match=re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n?",patchlines[i])
        assert match,patchlines[i]
        old_start=int(match.group(1))-1
        old_count=int(match.group(2) or 1)
        new_count=int(match.group(4) or 1)
        output.extend(lines[position:old_start]);position=old_start;i+=1
        old_seen=new_seen=0
        while i<len(patchlines) and not patchlines[i].startswith("@@ "):
            line=patchlines[i];i+=1
            tag,value=line[:1],line[1:]
            assert tag in [" ","+","-"]
            if tag in [" ","-"]:
                assert lines[position]==value,(position,lines[position],value)
                position+=1;old_seen+=1
            if tag in [" ","+"]:output.append(value);new_seen+=1
        assert (old_seen,new_seen)==(old_count,new_count)
    output.extend(lines[position:])
    return "".join(output)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--allow-unsealed",action="store_true",help="For package construction only; do not call this final manifest validation.")
    args=parser.parse_args()
    checks={}
    summary=load("case_summary.json")
    fatal=read("evidence/failure_traceback.log")
    assert "RayTaskError(ZMQError)" in fatal and "zmq.error.ZMQError" in fatal
    path=re.search(r'ipc path "([^"]+)"',fatal).group(1)
    assert len(path.encode())==summary["failed_socket_path_bytes"]==150
    assert "longer than 107" in fatal
    checks["original_failure_traceback_and_length"]=True
    old=read("code_context/before/continuous_driver.py");new=read("code_context/after/continuous_driver.py")
    before=forwarded(old);after=forwarded(new)
    assert tuple(k for k in before if k not in ["TMPDIR","TEMP","TMP"])==after
    assert set(before)-set(after)=={"TMPDIR","TEMP","TMP"}
    checks["source_derived_remote_forwarding_contract"]=True
    for name,patchfile in [("continuous_driver.py","01_runtime_env_tmpdir.patch"),("handoff_supervisor.py","02_recovery_binding.patch")]:
        beforetext=read("code_context/before/"+name)
        aftertext=read("code_context/after/"+name)
        ast.parse(beforetext);ast.parse(aftertext)
        assert apply_patch(beforetext,read("patches/"+patchfile))==aftertext
    checks["both_patches_reconstruct_after_snapshots"]=True
    cpu=load("evidence/historical_cpu_acceptance.json")
    assert cpu["passed"] is True and cpu["test_count"]==45
    assert "Ran 45 tests" in read("evidence/historical_cpu_tests.log") and read("evidence/historical_cpu_tests.log").rstrip().endswith("OK")
    ipc=load("evidence/eight_node_cpu_zmq.json")
    assert ipc["passed"] is True and len(ipc["rows"])==8
    assert all(r["actual_zmq_bind_passed"] is True and r["socket_path_bytes"] in [73,74] for r in ipc["rows"])
    assert ipc["runtime_env_temp_overrides"]==[]
    checks["historical_cpu_and_eight_node_records_consistent"]=True
    restore=load("evidence/recovery_restore_and_publication.json")
    assert restore["passed"] is True and restore["source_step"]==15
    assert len(restore["actual_restored_ranks"])==16
    assert all(r["rng_matches_actual_source"] is True and r["optimizer_steps"]["minimum"]==15 and r["optimizer_steps"]["maximum"]==15 for r in restore["actual_restored_ranks"])
    assert len(restore["actual_installed_replicas"])==12
    assert all(r["global_steps"]==15 for r in restore["actual_installed_replicas"])
    checks["historical_restore_and_initial_publication_consistent"]=True
    rows=(ROOT/"metrics/recovery_steps_16_17.jsonl").read_bytes().splitlines(keepends=True)
    assert [json.loads(line)["step"] for line in rows]==[16,17]
    for line in rows:
        j=json.loads(line);step=j["step"];receipt=load(f"evidence/recovery_step_{step}.json")
        assert any(digest(b)==receipt["metrics_row_sha256"] for b in [line,line.rstrip(b"\r\n")])
        assert receipt["passed"] is True and receipt["sampled_policy"]==step-1 and receipt["published_policy"]==step
        assert receipt["batch"]["nonpadding_rows"]==128 and receipt["batch"]["synthetic_padding_rows"]==0
        assert receipt["checkpoint"]["world_size"]==16 and receipt["checkpoint"]["global_step"]==step
        assert len(receipt["optimizer_steps_by_rank"])==16
        assert all(r["minimum"]==step and r["maximum"]==step for r in receipt["optimizer_steps_by_rank"].values())
        assert len(receipt["installed_versions"])==12 and all(r["global_steps"]==step for r in receipt["installed_versions"])
        value=j["data"]["actor/rollout_corr/token_logp_abs_diff"]
        assert isinstance(value,(int,float)) and math.isfinite(value) and value>0
    checks["original_metric_row_hashes_and_updates_16_17_consistent"]=True
    for entry in load("provenance/source_manifest.json")["files"]:
        assert digest((ROOT/entry["exported_path"]).read_bytes())==entry["export_sha256"]
    checks["source_to_export_manifest_hashes"]=True
    manifest=ROOT/"MANIFEST.json"
    if manifest.exists():
        manifest_data=load("MANIFEST.json")
        for entry in manifest_data["files"]:
            p=ROOT/entry["path"]
            assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
            data=p.read_bytes()
            assert len(data)==entry["bytes"] and digest(data)==entry["sha256"],entry["path"]
        checks["sealed_manifest_all_files"]=True
    else:
        assert args.allow_unsealed,"MANIFEST.json missing; this is not a sealed package."
        checks["sealed_manifest_all_files"]="not yet sealed; construction check only"
    result={"passed":True,"scope":"offline exported-package checks only; no RL/GPU or official AutoRLBench validator execution","checks":checks,"formal_benchmark_ready":False}
    encoded=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded,encoding="utf-8")
    print(encoded,end="")
if __name__=="__main__":main()
