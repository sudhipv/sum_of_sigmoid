"""Frozen feature proposals and training-only availability, before any fitting."""
from .frozen_detector import detect
from .legacy_consolidation import consolidate
from .turn_consolidation import consolidate_turns

def prepare(observations,cutoff,procedure='legacy'):
    raw=detect(observations[:cutoff+1].copy(),cutoff)
    legacy=consolidate(raw['flags'],raw['reversal_veto_events'],raw['steepening_veto_runs'],raw['pending_steepening_vetoes'],cutoff)
    # Exact-date groups preserve every raw label. Frozen legacy consolidation
    # performs association over raw flags and records before/after groups.
    if procedure=='legacy':
        groups=legacy['after'];alternative=None
    elif procedure=='turn_optional':
        alternative=consolidate_turns(raw,legacy);groups=alternative['finalized']
    else:raise ValueError('Unknown candidate procedure')
    return dict(raw=raw,identical_date_groups=legacy['before'],legacy=legacy,optional_turn=alternative,
                groups=groups,locations=sorted(g['date'] for g in groups),procedure=procedure)
