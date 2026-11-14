"""Operational gates for stated boundary-improvement claims on supplied counts."""
from itertools import combinations

import numpy as np

from rscd_metrics import bf1_scores, region_scores

FIELDS = ('pred_matches', 'gt_matches', 'pred_boundary', 'gt_boundary')
EPS = 1e-12


def assess_boundary_claims(frame_models):
    """Require strict gains for every requested frame/tolerance and deletion."""
    result = {}
    first = next(iter(frame_models.values()))
    for left, right in combinations(first, 2):
        values = []
        missing = 0
        checked = 0
        for models in frame_models.values():
            assert models[left].keys() == models[right].keys()
            keys = list(models[left])
            arrays = []
            for name in (left, right):
                arrays.append(np.asarray([[
                    [models[name][k]['boundary'][t][field] for field in FIELDS]
                    for t in ('0.5', '1.0', '2.0')] for k in keys], dtype=np.int64))
            totals = [a.sum(axis=0) for a in arrays]
            for omitted in [None, *range(len(keys))]:
                if omitted is not None and len(keys) == 1:
                    continue
                c = [total if omitted is None else total-a[omitted] for total,a in zip(totals, arrays)]
                for a,b in zip(*c):
                    x,y = [bf1_scores(dict(zip(FIELDS, map(int, row))))['bf1'] for row in (a,b)]
                    checked += 1
                    if x is None or y is None:
                        missing += 1
                    else:
                        values.append(y-x)
        guards = []
        for models in frame_models.values():
            def sums(name, kind):
                rows = list(models[name].values())
                return {k:sum(r[kind][k] for r in rows) for k in rows[0][kind]}
            a,b = [sums(name, 'region') for name in (left,right)]
            ia,ib = [region_scores(c)['iou'] for c in (a,b)]
            oa,ob = [sums(name,'objects') for name in (left,right)]
            guards.append(dict(iou_non_decreasing=None not in (ia,ib) and ib >= ia-EPS,
                               matched_not_less=ob['matched'] >= oa['matched'],
                               unmatched_predictions_not_more=ob['unmatched_pred'] <= oa['unmatched_pred']))
        passed = bool(values) and missing == 0 and min(values) > EPS
        result[f'{left} → {right}'] = dict(
            scope='pooled BF1; all requested frames and tolerances; full set and each single-sample omission',
            paired_conditions=checked, undefined=missing,
            minimum_gain_pp=100*min(values) if values else None,
            maximum_gain_pp=100*max(values) if values else None,
            boundary_direction_passed=passed, detection_guards=guards,
            joint_improvement_passed=passed and all(all(g.values()) for g in guards),
            margin_note='Zero is a direction threshold. Practical significance requires an external task-specific margin.')
    return result
