"""Run the submitted resolver out of the training process; return only a path."""
import importlib.util
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout


def main():
    script=Path(sys.argv[1]).resolve()
    try:
        request=json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            spec=importlib.util.spec_from_file_location('submission',script)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            value=module.resolve(request['root'],request['record'])
        if not isinstance(value,str):raise TypeError('resolve must return a relative path string')
        result={'ok':True,'path':value}
    except FileNotFoundError as error:
        result={'ok':False,'kind':'missing','message':str(error)[:2000]}
    except Exception as error:
        result={'ok':False,'kind':'error','type':type(error).__name__,'message':str(error)[:2000]}
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
