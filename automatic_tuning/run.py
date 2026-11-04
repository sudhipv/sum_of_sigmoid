"""Single public entry point. Preparation/ML is default; sampling is explicit."""
import argparse,json,os,sys
sys.dont_write_bytecode=True

# Preserve the original runner's shared seed when spawn imports this CLI.
# This is deliberately not a distinct-stream initializer. Sampling sets this
# internal environment value before creating the original sampler's Pool.
if 'SUMSIGMOID_ORIGINAL_TMCMC_SEED' in os.environ:
    import numpy as np
    np.random.seed(int(os.environ['SUMSIGMOID_ORIGINAL_TMCMC_SEED']))
    from sumsigmoid.mutation_controls import restore_worker_module
    restore_worker_module()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare');prep.add_argument('--config',required=True);prep.add_argument('--output',help='New output path,relative to current working directory');prep.add_argument('--tmcmc',action='store_true',help='Explicitly sample only after accepted ML')
    sample=commands.add_parser('tmcmc');sample.add_argument('--prepared',required=True);sample.add_argument('--dry-run',action='store_true')
    posterior=commands.add_parser('posterior',help='Evaluate saved samples only; no optimization or sampling')
    posterior.add_argument('--inference',help='tmcmc/ output with immutable run configuration and named samples')
    posterior.add_argument('--configuration',help='Exported run_configuration.json, alternative to --inference')
    posterior.add_argument('--samples',help='Named posterior_samples.csv with --configuration')
    posterior.add_argument('--data',required=True,help='Daily day,I,is_observed CSV, days0..199')
    posterior.add_argument('--truth',help='Synthetic generating_curves.npz; omit for Toronto')
    posterior.add_argument('--output',required=True)
    posterior.add_argument('--reference',help='Optional compact accepted reference directory')
    posterior.add_argument('--draw-id',type=int,help='One saved row instead of the full ensemble')
    posterior.add_argument('--noise-seed',type=int,default=7106,help='Observation-noise draws only, not inference RNG')
    args=parser.parse_args()
    if args.command=='prepare':
        from sumsigmoid.workflow import run
        result=run(args.config,args.tmcmc,args.output)
    elif args.command=='tmcmc':
        from sumsigmoid.optional_tmcmc import run
        result=run(args.prepared,args.dry_run)
    else:
        from pathlib import Path
        from sumsigmoid.posterior import export_inference_inputs,regenerate
        if bool(args.inference)==bool(args.configuration):
            parser.error('Choose exactly one of --inference or --configuration')
        if args.inference:
            inputs=export_inference_inputs(args.inference,args.data,args.truth,Path(args.output)/'inputs',args.noise_seed)
            config,samples,data=inputs/'run_configuration.json',inputs/'posterior_samples.csv',inputs/'observed_daily.csv'
            truth=inputs/'generating_curves.npz' if args.truth else None
        else:
            if not args.samples:parser.error('--configuration requires --samples')
            config,samples,data,truth=args.configuration,args.samples,args.data,args.truth
        result=regenerate(config,samples,data,truth,args.output,args.draw_id,args.reference)
    print(json.dumps(result,indent=2))
    return 2 if result.get('status')=='unresolved' else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);raise SystemExit(1)
