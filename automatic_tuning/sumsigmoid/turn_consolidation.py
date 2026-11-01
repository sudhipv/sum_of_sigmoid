"""Version 2: a single, turn-anchored association layer over legacy groups.

No detections, fitting, model evaluations, amplitude signs, spacing or chained
support overlap enter this module. All dates and thresholds are caller-supplied
detector evidence, apart from the explicitly approved comparison stencils and
availability guard. No source record is mutated.
"""
from copy import deepcopy


TURN_LABELS = {'observed peak', 'observed trough'}


def reversal_id(event):
    """Stable ID from the detector's existing reversal onset and directions."""
    return f"reversal:{event['dated_at_day']}:{event['from']}:{event['to']}"


def group_key(flags):
    return '|'.join(sorted(f['id'] for f in flags))


def comparison_interval(flag):
    if flag['label'] == 'sustained flattening':
        return [flag['date']-5, flag['date']+2]
    if flag['label'] == 'decline slowing':
        return [flag['date']-5, flag['date']+5]
    return None


def compatible(flag, anchor):
    if flag['label'] == 'decline slowing':
        return anchor['label'] == 'observed trough'
    if flag['label'] == 'sustained flattening':
        previous = flag['source_event']['from']
        return ((previous == 'rising' and anchor['label'] == 'observed peak')
                or (previous == 'falling' and anchor['label'] == 'observed trough'))
    return False


def legacy_group_available(group, legacy):
    times = [f['available_by_raw_day'] for f in group['flags']]
    ids = {f['id'] for f in group['flags']}
    for row in legacy['audit']:
        if (row.get('status') == 'consolidated'
                and {row['flattening_id'], row['slowing_id']} <= ids):
            times.append(row['required_raw_day'])
    return max(times)


def consolidate_turns(raw, legacy):
    """Apply the approved page-6 association exactly once at this raw cutoff.

    `raw` must come from the unchanged detector using only that prefix. `legacy`
    must be that prefix's unchanged legacy consolidation, never full-window
    groups injected into an earlier prefix. Finalized output requires complete
    source evidence and c>=h+10. The same guard is reported for standalone output
    decisions; their original source availability remains separately recorded.
    Finalized-at-cutoff does not mean immune to new evidence in later prefixes.
    """
    c = raw['cutoff']
    assert legacy['cutoff'] == c
    assert raw['last_observation_used'] == c
    groups = deepcopy(legacy['after'])
    anchors = []
    for group in groups:
        local_turns = [f for f in group['flags'] if f['label'] in TURN_LABELS]
        if len(local_turns) > 1:
            raise RuntimeError('POLICY GAP: legacy exact-date group contains distinct turn anchors; do not silently combine or split them')
        for flag in local_turns:
            event = flag['source_event']
            anchors.append(dict(id=reversal_id(event), flag_id=flag['id'],
                label=flag['label'], date=flag['date'],
                confirmation_day=flag['confirmation']['confirmed_on_daily_check'],
                available_by_raw_day=flag['available_by_raw_day'],
                reversal=deepcopy(event), old_group=group_key(group['flags'])))
    assert len({a['id'] for a in anchors}) == len(anchors)
    anchor_by_id = {a['id']:a for a in anchors}
    anchor_group_ids = {a['old_group']:a['id'] for a in anchors}
    audit = []
    destinations = {}
    # Anchor destinations are stable; evaluating another group never adds anchors.
    for group in groups:
        key = group_key(group['flags'])
        available = legacy_group_available(group, legacy)
        row = dict(old_group=key, old_representative=group['date'],
                   member_ids=sorted(f['id'] for f in group['flags']),
                   legacy_available=available, nominations=[],
                   nomination_anchors=[], pending_turn_evidence=[], veto_checks=[], status=None, reason=None)
        if key in anchor_group_ids:
            anchor_id = anchor_group_ids[key]
            row.update(status='anchor', anchor_id=anchor_id,
                       reason='Existing confirmed turn remains its own episode anchor')
            destinations[key] = 'turn:'+anchor_id
        elif any(f['label']=='decline steepening' for f in group['flags']):
            row.update(status='independent', reason='Sustained-steepening groups remain independent')
            destinations[key] = 'legacy:'+key
        else:
            for f in group['flags']:
                span = comparison_interval(f)
                if span is None:
                    continue
                for anchor in anchors:
                    if compatible(f, anchor) and span[0] <= anchor['date'] <= span[1]:
                        row['nominations'].append(dict(member_id=f['id'],
                            comparison_interval=span, anchor_id=anchor['id'],
                            extremum_date=anchor['date']))
                # A not-yet-final turn is never an anchor. If its provisional
                # evidence conflicts with the same bracket, report/defer it.
                for flag in raw['flags']:
                    if (flag['label'] in TURN_LABELS and flag['available_by_raw_day']>c
                            and compatible(f,flag) and span[0]<=flag['date']<=span[1]):
                        row['pending_turn_evidence'].append(dict(member_id=f['id'],
                            flag_id=flag['id'],reversal_id=reversal_id(flag['source_event']),
                            provisional_extremum=flag['date'],
                            provisional_raw_availability=flag['available_by_raw_day']))
            nominated = sorted({n['anchor_id'] for n in row['nominations']})
            row['nomination_anchors'] = nominated
            destinations[key] = 'legacy:'+key
            if not nominated:
                row.update(status='deferred' if row['pending_turn_evidence'] else 'unmatched',
                    reason='Compatible provisional turn evidence is not finalized' if row['pending_turn_evidence'] else 'No compatible finalized turn extremum lies in a member comparison interval at this cutoff')
            elif len(nominated)>1:
                row.update(status='ambiguous', reason='Members nominate more than one distinct compatible confirmed turn; no arbitrary choice')
            else:
                anchor = anchor_by_id[nominated[0]]
                # A nominally unique anchor is not enough: guard all original
                # members, not just the member that nominated this anchor.
                anchor_group = next(g for g in groups if group_key(g['flags'])==anchor['old_group'])
                h=max([f['date'] for f in group['flags']+anchor_group['flags']]+[anchor['date']])
                need=max(available,legacy_group_available(anchor_group,legacy),h+10)
                row.update(anchor_id=anchor['id'], latest_representative=h,
                           guard_cutoff=h+10, required_cutoff=need)
                for f in group['flags']:
                    lo,hi=sorted([f['date'],anchor['date']])
                    reversals=[deepcopy(e) for e in raw['reversal_veto_events']
                        if e['from'] is not None and reversal_id(e)!=anchor['id']
                        and lo<=e['dated_at_day']<=hi and e['available_by_raw_day']<=c]
                    steepening=[deepcopy(e) for e in raw['steepening_veto_runs']
                        if e['interval'][0]<=hi and e['interval'][1]>=lo
                        and e['confirmed_available_by_raw_day']<=c]
                    pending=[deepcopy(e) for e in raw['pending_steepening_vetoes']
                        if e['interval'][0]<=hi and e['interval'][1]>=lo]
                    row['veto_checks'].append(dict(member_id=f['id'],interval=[lo,hi],
                         reversal_vetoes=reversals,steepening_vetoes=steepening,pending_vetoes=pending))
                if any(v['reversal_vetoes'] or v['steepening_vetoes'] for v in row['veto_checks']):
                    row.update(status='blocked',reason='Another confirmed reversal or sustained steepening separates a member and the nominated anchor')
                elif row['pending_turn_evidence']:
                    row.update(status='deferred',reason='Other compatible provisional turn evidence has not finalized')
                elif any(v['pending_vetoes'] for v in row['veto_checks']):
                    row.update(status='deferred',reason='Relevant pending steepening evidence has not resolved')
                elif need>c:
                    row.update(status='deferred',reason='Source/anchor availability or the approved h+10 guard is not satisfied')
                else:
                    row.update(status='associated',reason='Unique compatible turn inside comparison evidence; all original members pass veto and availability checks')
                    destinations[key]='turn:'+anchor['id']
        row['destination']=destinations[key]
        audit.append(row)
    # Gather once by the chosen anchor; no graph closure, overlap chaining,
    # proximity grouping or date-based post-merge is performed.
    gathered={}
    for group in groups:
        key=group_key(group['flags']);dest=destinations[key]
        if dest not in gathered:
            anchor_id=dest.removeprefix('turn:') if dest.startswith('turn:') else None
            gathered[dest]=dict(id=dest,anchor_id=anchor_id,
                date=anchor_by_id[anchor_id]['date'] if anchor_id else group['date'],
                old_groups=[],flags=[],old_availability=[])
        gathered[dest]['old_groups'].append(key)
        gathered[dest]['flags'].extend(deepcopy(group['flags']))
        gathered[dest]['old_availability'].append(legacy_group_available(group,legacy))
    finalized=[];provisional=[]
    for group in gathered.values():
        group['flags'].sort(key=lambda f:(f['date'],f['id']))
        group['old_groups'].sort()
        group['original_dates']=sorted({f['date'] for f in group['flags']})
        group['labels']=[f['label'] for f in group['flags']]
        group['member_ids']=sorted(f['id'] for f in group['flags'])
        group['source_available']=max(f['available_by_raw_day'] for f in group['flags'])
        h=max([f['date'] for f in group['flags']]+[group['date']])
        group.update(latest_representative=h, guard_cutoff=h+10,
                     required_cutoff=max(group['old_availability']+[h+10]))
        unresolved=any(a['status']=='deferred' for a in audit if a['old_group'] in group['old_groups'])
        if group['required_cutoff']>c or unresolved:
            group.update(status='provisional',reason='Output guard or a possible association remains pending; not eligible as a finalized candidate')
            provisional.append(group)
        else:
            group.update(status='finalized_at_cutoff',reason='All currently participating evidence and guard finalized at this cutoff; future prefixes may supply new features')
            finalized.append(group)
    all_output=finalized+provisional
    assert sorted(f['id'] for g in all_output for f in g['flags']) == sorted(f['id'] for g in groups for f in g['flags'])
    for anchor in anchors:
        assert sum(anchor['flag_id'] in g['member_ids'] for g in all_output)==1
    return dict(version='turn_association_v2',cutoff=c,last_observation_used=c,
                anchors=anchors,audit=audit,
                finalized=sorted(finalized,key=lambda g:(g['date'],g['id'])),
                provisional=sorted(provisional,key=lambda g:(g['date'],g['id'])),
                availability_policy='Raw/legacy availability preserved separately; h+10 applied to every emitted finalized output decision, including unchanged singleton groups')
