"""Associate existing detector events; no detection thresholds or fitting."""
from copy import deepcopy


def consolidate(flags, reversals, steepening, pending_steepening, cutoff):
    """Flattening onset inside/on closure of a slowing run, without vetoes.

    The intervening interval is closed: slowing-run start to flattening onset.
    Steepening intervals overlapping it veto association. This defines temporal
    association only; no distance threshold or new detector is introduced.
    """
    available=[deepcopy(f) for f in flags if f['available_by_raw_day']<=cutoff]
    destinations={f['id']:f['date'] for f in available}
    slowing=[f for f in available if f['label']=='decline slowing']
    audit=[]
    for flat in available:
        if flat['label']!='sustained flattening':
            continue
        f=flat['date']
        matched=[]
        for slow in slowing:
            a,b=slow['qualifying_interval']
            if not a<=f<=b+1:
                continue
            record={'flattening_id':flat['id'],'slowing_id':slow['id'],
                    'original_dates':[slow['date'],f],'slowing_interval':[a,b],
                    'flattening_confirmation_interval':flat['qualifying_interval'],
                    'intervening_interval':[a,f],
                    'required_raw_day':max(slow['available_by_raw_day'],flat['available_by_raw_day'],f+8,f+5),
                    'reversal_vetoes':[],'steepening_vetoes':[],'pending_vetoes':[]}
            if record['required_raw_day']>cutoff:
                record.update(status='deferred',reason='Complete local slope/reversal evidence not available at this cutoff')
            else:
                record['reversal_vetoes']=[deepcopy(r) for r in reversals
                    if r['from'] is not None and r['from']!=r['to']
                    and a<=r['dated_at_day']<=f and r['available_by_raw_day']<=cutoff]
                record['steepening_vetoes']=[deepcopy(r) for r in steepening
                    if r['interval'][0]<=f and r['interval'][1]>=a and r['confirmed_available_by_raw_day']<=cutoff]
                record['pending_vetoes']=[deepcopy(r) for r in pending_steepening
                    if r['interval'][0]<=f and r['interval'][1]>=a]
                if record['reversal_vetoes'] or record['steepening_vetoes']:
                    record.update(status='blocked',reason='Existing confirmed reversal or sustained steepening intervenes')
                elif record['pending_vetoes']:
                    record.update(status='deferred',reason='A possible sustained steepening has not resolved')
                else:
                    record.update(status='eligible',reason='Event membership/adjacent closure; no intervening veto')
                    matched.append((slow,record))
            audit.append(record)
        if len(matched)==1:
            slow,record=matched[0]
            destinations[flat['id']]=slow['date']
            record.update(status='consolidated',representative_date=slow['date'],
                          representative_rule='existing maximum slowing contrast; earliest date on ties',
                          member_flags=[deepcopy(slow),deepcopy(flat)])
        elif len(matched)>1:
            for _,record in matched:
                record.update(status='ambiguous',reason='More than one eligible run; no arbitrary representative selected')
        elif not any(r['flattening_id']==flat['id'] for r in audit):
            audit.append({'flattening_id':flat['id'],'original_dates':[f],
                          'status':'unmatched','reason':'No slowing run contains or is directly closed by this onset'})
    before={}
    after={}
    for flag in available:
        before.setdefault(flag['date'],[]).append(deepcopy(flag))
        after.setdefault(destinations[flag['id']],[]).append(deepcopy(flag))
    def groups(table):
        return [{'date':day,'original_dates':sorted(set(f['date'] for f in members)),
                 'labels':[f['label'] for f in members],'flags':members,
                 'source_available_by_raw_day':max(f['available_by_raw_day'] for f in members)}
                for day,members in sorted(table.items())]
    return {'cutoff':cutoff,'before':groups(before),'after':groups(after),'audit':audit}
