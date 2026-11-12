#!/usr/bin/env python3
"""Execute the independent software tests and the strict provenance readiness check."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'RGR_Net_Executable_Reproducibility'
steps=[(sys.executable,['-m','pytest','-q',str(CORE/'tests')]),(sys.executable,[str(CORE/'scripts'/'run_readiness.py'),'--strict'])]
for exe,args in steps:
    r=subprocess.run([exe,*args],cwd=CORE)
    print('exit_code',r.returncode,'command',exe,*args)
    if '--strict' not in args and r.returncode:sys.exit(r.returncode)
print('Software tests and readiness check completed. Strict readiness may correctly report evidence incomplete.')
