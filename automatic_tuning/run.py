"""Single public entry point. Preparation/ML is default; sampling is explicit."""
import argparse,json,os,sys
sys.dont_write_bytecode=True

# Preserve the original runner's shared seed when spawn imports this CLI.
# This is deliberately not a distinct-stream initializer. Sampling sets this
# internal environment value before creating the original sampler's Pool.
if 'SUMSIGMOID_ORIGINAL_TMCMC_SEED' in os.environ:
    import numpy as np
    np.random.seed(int(os.environ['SUMSIGMOID_ORIGINAL_TMCMC_SEED']))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--config',required=True);prep.add_argument('--output',help='New output path,relative to current working directory');prep.add_argument('--tmcmc',action='store_true',help='Explicitly sample only after accepted ML')
    sample=commands.add_parser('tmcmc');sample.add_argument('--prepared',required=True);sample.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        from sumsigmoid.workflow import run
        result=run(args.config,args.tmcmc,args.output)
    else:
        from sumsigmoid.optional_tmcmc import run
        result=run(args.prepared,args.dry_run)
    print(json.dumps(result,indent=2))
    return 2 if result.get('status')=='unresolved' else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);raise SystemExit(1)
