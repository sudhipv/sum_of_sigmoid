"""Calculate and plot trajectories from existing posterior samples. No inference."""
from pathlib import Path
import argparse, time, json, sys
from posterior_forward import calculate, case_ids
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--case',help='A supplied ID, or all (default)')
    group.add_argument('--config',type=Path,help='Matching alternative configuration.json')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--samples',type=Path,help='Compatible named CSV / labelled NPZ with explicit weights; one configuration only')
    parser.add_argument('--conditioning',choices=['auto','none','supplied'],default='auto')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.list:print('\n'.join(case_ids(ROOT)));return 0
    selected=case_ids(ROOT) if not args.config and args.case in [None,'all'] else [args.case or 'custom']
    if args.samples is not None and len(selected)>1:raise ValueError('--samples requires a single --case or --config.')
    root=args.output or ROOT/'outputs'/f'forward_{time.time_ns()}'
    exitcode=0
    for cid in selected:
        if args.config is None and cid not in case_ids(ROOT):raise ValueError('Unknown case. Run python run.py --list.')
        config=args.config if args.config is not None else ROOT/'cases'/cid/'configuration.json'
        print(f'Calculating {cid} from posterior samples',flush=True)
        info,figure,_=calculate(config,root/cid,args.samples,args.conditioning)
        if figure is not None:
            import matplotlib.pyplot as plt
            plt.close(figure)
        print(json.dumps(info,indent=2))
        if info['status']!='PASS':exitcode=2
    return exitcode

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError)as error:
        print('Input or validity error: '+str(error),file=sys.stderr);raise SystemExit(2)
