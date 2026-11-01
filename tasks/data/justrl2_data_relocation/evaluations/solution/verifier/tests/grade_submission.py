#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Grade only the resumed run against the immutable prepared incident."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    results=Path(os.environ.get('AUTORLBENCH_RESULTS','/results'))
    prepared=Path(os.environ.get('AUTORLBENCH_PREPARED','/autorlbench/prepared'))
    resources=Path(__file__).resolve().parent.parent/'resources'
    try:
        path=resources/'incident_audit/verify_resume.py'
        spec=importlib.util.spec_from_file_location('_hidden_incident_resume',path)
        engine=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        pin=json.loads((resources/'snapshot.json').read_text())
        scene=prepared/'incident'
        if hashlib.sha256((scene/'SCENE.json').read_bytes()).hexdigest()!=pin['scene_sha256']:
            raise ValueError('Scene pin mismatch')
        proof=engine.verify_resume(scene,results/'experiment_logs/run-home',fresh=True)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        proof={'schema_version':3,'score':False,'errors':[str(error)]}
    result={'passed':proof['score'] is True,'score':int(proof['score'] is True),'metrics':proof}
    results.mkdir(parents=True,exist_ok=True)
    (results/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (results/'metrics.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
