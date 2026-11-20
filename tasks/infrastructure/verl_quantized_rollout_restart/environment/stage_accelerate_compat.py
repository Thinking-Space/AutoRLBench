"""Stage source-verified Accelerate Python compatibility overlay. codex."""
import argparse, hashlib, importlib.util, pathlib, shutil
p=argparse.ArgumentParser();p.add_argument('--target',required=True);a=p.parse_args()
spec=importlib.util.find_spec('accelerate');src=pathlib.Path(spec.origin).parent; original=src/'big_modeling.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(original)=='b38df85d2e081ef89d5117b3fa238a029cb95750e462686cb474abdcb681ea5a'
dest=pathlib.Path(a.target)/'accelerate';assert not dest.exists();shutil.copytree(src,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=dest/'big_modeling.py';s=p.read_text();old='''            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)''';new='''            kwargs = dict(module._parameters[name].__dict__)
            hf_initialized = kwargs.pop("_is_hf_initialized", None)
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            if hf_initialized is not None:
                module._parameters[name]._is_hf_initialized = hf_initialized'''
assert s.count(old)==1;p.write_text(s.replace(old,new,1));print('original_sha256='+sha(original)+'\noverlay_sha256='+sha(p))
