"""CPU-only Unix-socket length mechanism check; no RL or external dependencies."""
import argparse
import json
import os
from pathlib import Path
import socket
import tempfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,default=Path(__file__).resolve().parent/".runtime")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or os.environ.get("NO_GPU")!="1":
        parser.error('Run with CUDA_VISIBLE_DEVICES="" NO_GPU=1.')
    if not hasattr(socket,"AF_UNIX"):
        parser.error("This mechanism check requires Unix domain sockets.")
    work=args.work_dir.resolve()
    work.mkdir(parents=True,exist_ok=True)
    original_cwd=Path.cwd()
    report={"scope":"CPU socket mechanism only; not Ray/vLLM or RL acceptance","gpu_operations":0,"training_jobs_started":0}
    with tempfile.TemporaryDirectory(prefix="ipc-check-",dir=str(work)) as tmp:
        directory=Path(tmp)
        # Ensure the absolute endpoint exceeds the common Linux 107-byte usable limit.
        extra=max(1,120-len(os.fsencode(str(directory))))
        directory=directory/("d"*extra)
        directory.mkdir()
        target=directory/"endpoint.sock"
        assert target.resolve().is_relative_to(work)
        report["absolute_endpoint_bytes"]=len(os.fsencode(str(target)))
        assert report["absolute_endpoint_bytes"]>107
        sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            try:
                sock.bind(str(target))
            except OSError as error:
                report["absolute_bind_rejected"]=True
                report["absolute_bind_error_type"]=type(error).__name__
                report["absolute_bind_error"]=str(error)
            else:
                raise AssertionError("Long absolute path unexpectedly bound; do not claim mechanism reproduced.")
        finally:
            sock.close()
        assert not target.exists(),"Long-path bind unexpectedly created an endpoint."
        try:
            os.chdir(directory)
            sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            try:
                sock.bind("endpoint.sock")
                assert target.is_socket()
                report["short_relative_endpoint_bytes"]=len(b"endpoint.sock")
                report["short_bind_passed"]=True
                report["same_owned_directory"]=True
            finally:
                sock.close()
                if target.exists():target.unlink()
        finally:
            os.chdir(original_cwd)
    report["temporary_socket_cleaned"]=True
    report["passed"]=all(report.get(k) is True for k in ["absolute_bind_rejected","short_bind_passed","same_owned_directory","temporary_socket_cleaned"])
    encoded=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded,encoding="utf-8")
    print(encoded,end="")
if __name__=="__main__":main()
