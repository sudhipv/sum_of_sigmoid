"""Pure read-only adapter of existing detector formulas for cutoff replay.

Full-cutoff output is checked against preserved raw outputs by the runner.
Availability metadata distinguishes feature dates from finalized evidence.
No original detector files are imported/executed or overwritten.
"""
import numpy as np


def detect(observations, cutoff):
    y=np.asarray(observations[:cutoff+1],dtype=float).copy()
    if len(y)!=cutoff+1 or cutoff>159:
        raise ValueError('Only declared training observations may be used')
    n=len(y)
    s=np.array([y[max(0,t-3):min(n,t+4)].mean() for t in range(n)])
    relative=np.full(n,np.nan)
    relative[5:]=(s[5:]-s[:-5])/np.maximum(s[:-5],1)
    names={-1:'falling',0:'within +/-15%',1:'rising'}
    raw=np.where(relative>=.15,1,np.where(relative<=-.15,-1,0))
    categories=[]
    pending=confirmed=None
    run=0
    for t in range(5,n):
        value=int(raw[t]);run=run+1 if value==pending else 1;pending=value
        if run>=3 and value!=confirmed:
            categories.append({'confirmed_on_day':t,'dated_at_day':t-2,
                               'from':None if confirmed is None else names[confirmed],'to':names[value]})
            confirmed=value
    directions=[]
    pending=last=None
    run=0
    for t in range(5,n):
        value=int(np.sign(relative[t]))
        if value==0:
            pending=None;run=0;continue
        run=run+1 if value==pending else 1;pending=value
        if run>=3 and value!=last:
            directions.append({'confirmed_on_day':t,'dated_at_day':t-2,'from':last,'to':value})
            last=value
    flags=[]
    for e in categories:
        if e['to']!='within +/-15%':
            continue
        d=e['dated_at_day'];c=e['confirmed_on_day']
        flags.append({'id':f'flattening:{d}','date':d,'label':'sustained flattening',
                      'qualifying_interval':[d,c],'confirmation':{'confirmed_on_daily_check':c,
                          'raw_data_through':c+3,'dated_at':d},
                      'available_by_raw_day':c+3,'source_event':e})
    for i,e in enumerate(directions):
        if e['from'] is None:
            continue
        left=max(0,directions[i-1]['dated_at_day']-5)
        right=e['confirmed_on_day']
        peak=e['from']==1 and e['to']==-1
        d=left+int(np.argmax(s[left:right+1]) if peak else np.argmin(s[left:right+1]))
        label='observed peak' if peak else 'observed trough'
        flags.append({'id':f'{label}:{d}','date':d,'label':label,
                      'qualifying_interval':[e['dated_at_day'],right],
                      'extremum_search_interval':[left,right],
                      'confirmation':{'confirmed_on_daily_check':right,'raw_data_through':right+3,
                                      'direction_onset':e['dated_at_day']},
                      'available_by_raw_day':right+3,'source_event':e})
    episodes=[]
    for i,e in enumerate(directions):
        if e['to']!=-1:
            continue
        rise=next((r for r in directions[i+1:] if r['to']==1),None)
        episodes.append({'start_day':max(0,e['dated_at_day']-5),
                         'declining_checks_start_day':e['dated_at_day'],
                         'end_day':rise['dated_at_day']-1 if rise else cutoff})
    rows=[]
    source_candidates=[]
    stable_steepening=[]
    pending_steepening=[]
    for index,episode in enumerate(episodes):
        a,b=episode['start_day'],episode['end_day']
        local=[]
        for t in range(a,b+1):
            anchor=float(np.max(s[a:t+1]));drop=(anchor-s[t])/anchor
            before=(s[t]-s[t-5])/5 if t>=5 else None
            after=(s[t+5]-s[t])/5 if t+5<n else None
            contrast=None;kind='not eligible'
            if before is not None and after is not None and before<0 and after<0:
                contrast=(abs(after)-abs(before))/abs(before)
                kind=('steepening' if contrast>=.3 else 'slowing' if contrast<=-.3 else 'below slope-change threshold') if drop>=.2 else 'below cumulative-decline trigger'
            row={'day':t,'episode':index+1,'smoothed_infections':float(s[t]),'running_peak':anchor,
                 'cumulative_decline_percent':float(100*drop),
                 'slope_before_counts_per_day':None if before is None else float(before),
                 'slope_after_counts_per_day':None if after is None else float(after),
                 'relative_decline_rate_change_percent':None if contrast is None else float(100*contrast),
                 'raw_condition':kind,'retained_candidate':False}
            local.append(row);rows.append(row)
        episode['first_twenty_percent_trigger_day']=next((r['day'] for r in local if r['cumulative_decline_percent']>=20),None)
        start=0
        for end in range(1,len(local)+1):
            if end<len(local) and local[end]['raw_condition']==local[start]['raw_condition']:
                continue
            group=local[start:end];kind=group[0]['raw_condition']
            if kind in ['slowing','steepening'] and len(group)>=3:
                best=max(group,key=lambda r:abs(r['relative_decline_rate_change_percent']))
                best['retained_candidate']=True
                first,last=group[0]['day'],group[-1]['day']
                candidate={'day':best['day'],'type':kind,'episode':index+1,'qualifying_run':[first,last],
                           'slope_before':best['slope_before_counts_per_day'],'slope_after':best['slope_after_counts_per_day'],
                           'cumulative_decline_percent':best['cumulative_decline_percent'],
                           'relative_decline_rate_change_percent':best['relative_decline_rate_change_percent']}
                source_candidates.append(candidate)
                flags.append({'id':f'{kind}:{first}:{last}:{best["day"]}','date':best['day'],
                              'label':'decline '+kind,'qualifying_interval':[first,last],
                              'confirmation':{'first_three_checks':[first,first+1,first+2],
                                  'confirmed_on_daily_check':first+2,'first_confirmation_raw_data_through':first+10,
                                  'closing_check':last+1,'representative_finalized_raw_data_through':last+9},
                              'available_by_raw_day':last+9,'source_event':candidate})
            start=end
        # Veto uses the existing>=3 steepening definition, not its representative.
        stable=[r for r in local if r['day']+8<=cutoff]
        start=0
        for end in range(1,len(stable)+1):
            if end<len(stable) and stable[end]['raw_condition']==stable[start]['raw_condition']:
                continue
            group=stable[start:end]
            if group[0]['raw_condition']=='steepening':
                first,last=group[0]['day'],group[-1]['day']
                if len(group)>=3:
                    stable_steepening.append({'interval':[first,last],'confirmed_on_daily_check':first+2,
                                             'confirmed_available_by_raw_day':first+10})
                elif last==cutoff-8:
                    pending_steepening.append({'interval':[first,last],
                                               'reason':'Still-open steepening run has fewer than three stable checks'})
            start=end
    return {'cutoff':cutoff,'last_observation_used':cutoff,'smoothing':s,'relative':relative,
            'category_events':categories,'direction_events':directions,'flags':flags,
            'episodes':episodes,'slope_candidates':source_candidates,'slope_daily_rows':rows,
            'steepening_veto_runs':stable_steepening,'pending_steepening_vetoes':pending_steepening,
            'reversal_veto_events':[{**e,'available_by_raw_day':e['confirmed_on_day']+3} for e in directions]}
