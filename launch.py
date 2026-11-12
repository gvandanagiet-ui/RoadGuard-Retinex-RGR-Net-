#!/usr/bin/env python3
"""CLI for the packaged evaluation and independent reference model.
Linux x86_64/WSL2 + CPython 3.13 required. Data/checkpoints not included.
`strict-readiness` fails if original RGR-Net experiment evidence is absent.
"""
import argparse,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parent
SP=ROOT/'linux_cp313_runtime'/'site_packages'
PROJECT=ROOT/'project'
CORE=PROJECT/'RGR_Net_Executable_Reproducibility'
SCRIPTS=CORE/'scripts'
REFERENCE=PROJECT/'INDEPENDENT_REFERENCE_NOT_ORIGINAL'/'reference_code'
FACTORIAL=PROJECT/'REPRODUCIBILITY_EXECUTION_SOURCE'/'run_reference_factorial.py'

def run(args):
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(SP),str(CORE)])
    env['PYTHONNOUSERSITE']='1'
    return subprocess.call([sys.executable,'-S',*args],cwd=PROJECT,env=env)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check','tests','readiness','strict-readiness','schedule','factorial','reference-train','quality','detector','benchmark','imports'])
    ns,extra=parser.parse_known_args()
    cmd=ns.command
    if cmd=='tests': return run(['-m','pytest','-q',str(CORE/'tests'),*extra])
    if cmd=='readiness': return run([str(SCRIPTS/'run_readiness.py'),*extra])
    if cmd=='strict-readiness': return run([str(SCRIPTS/'run_readiness.py'),'--strict',*extra])
    if cmd=='schedule': return run([str(SCRIPTS/'generate_training_schedule.py'),*extra])
    if cmd=='factorial':
        args=[str(FACTORIAL)]
        if '--out' not in extra: args+=['--out',str(PROJECT/'reference_factorial_registry')]
        if '--execute' not in extra and '--dry-run' not in extra:args+=['--dry-run']
        return run([*args,*extra])
    if cmd=='reference-train':
        extra=[{'--train-dir':'--train_dir','--val-dir':'--val_dir'}.get(x,x) for x in extra]
        return run([str(REFERENCE/'train_reference.py'),*extra] if extra else [str(REFERENCE/'train_reference.py'),'--help'])
    if cmd=='quality': return run([str(SCRIPTS/'evaluate_quality.py'),*extra] if extra else [str(SCRIPTS/'evaluate_quality.py'),'--help'])
    if cmd=='detector': return run([str(SCRIPTS/'evaluate_detector_coco.py'),*extra] if extra else [str(SCRIPTS/'evaluate_detector_coco.py'),'--help'])
    if cmd=='benchmark': return run([str(SCRIPTS/'benchmark_torchscript.py'),*extra] if extra else [str(SCRIPTS/'benchmark_torchscript.py'),'--help'])
    if cmd=='imports': return run(['-c','import numpy,scipy,torch,torchvision,PIL,skimage,pandas;print("DEPENDENCIES OK", numpy.__version__,scipy.__version__,torch.__version__,torchvision.__version__,pandas.__version__)'])
    if cmd=='self-check':
        checks=[['-m','pytest','-q',str(CORE/'tests')], [str(SCRIPTS/'generate_training_schedule.py')], [str(SCRIPTS/'run_readiness.py')], ['-c','import numpy,scipy,torch,torchvision,PIL,skimage,pandas;print("BUNDLED PYTHON RUNTIME OK")']]
        for test in checks:
            r=run(test)
            if r: return r
        print('SELF-CHECK PASSED. Authentic RGR-Net experiment evidence must be separately supplied.')
        return 0
    return 1
if __name__=='__main__':sys.exit(main())
