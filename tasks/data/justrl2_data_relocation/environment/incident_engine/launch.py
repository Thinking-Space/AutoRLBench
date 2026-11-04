"""Clean interpreter entry for the trusted continuation driver."""
import os
from pathlib import Path
import sys

# Invoked with -I -S: no user-site startup code has run. Only trusted engine
# modules are imported before run.py starts the pinned training interpreter.
sys.path.insert(0, str(Path(__file__).resolve().parent))
for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD',
             'BASH_ENV', 'ENV', 'CDPATH'):
    os.environ.pop(name, None)
os.environ['PYTHONNOUSERSITE'] = '1'
# Load installed packages only after discarding submission-controlled startup paths.
import site
site.main()
from run import main

if __name__ == '__main__': main()
