#!/usr/bin/env python3
"""Launch available RGR-Net evaluation utilities (not original training code)."""
import argparse, subprocess, sys, pathlib
ROOT=pathlib.Path(__file__).resolve().parent
PROGRAMS={
  "tests": ["-m","pytest","-q",str(ROOT/"RGR_Net_Executable_Reproducibility/tests")],
  "readiness": [str(ROOT/"RGR_Net_Executable_Reproducibility/scripts/run_readiness.py"),"--help"],
  "schedule": [str(ROOT/"RGR_Net_Executable_Reproducibility/scripts/generate_training_schedule.py"),"--help"],
  "reference-train": [str(ROOT/"INDEPENDENT_REFERENCE_NOT_ORIGINAL/reference_code/train_reference.py"),"--help"],
  "reference-factorial": [str(ROOT/"REPRODUCIBILITY_EXECUTION_SOURCE/run_reference_factorial.py"),"--help"],
  "quality": [str(ROOT/"RGR_Net_Executable_Reproducibility/scripts/evaluate_quality.py"),"--help"],
}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("command",choices=PROGRAMS)
 p.add_argument('args',nargs=argparse.REMAINDER)
 a=p.parse_args()
 cmd=PROGRAMS[a.command]
 if a.args:cmd=[item for item in cmd if item!='--help']+a.args
 raise SystemExit(subprocess.call([sys.executable]+cmd,cwd=ROOT))
if __name__=='__main__':main()
