"""Audit cohort selection by NHANES cycle, matching pipeline/step5_missing.py.

Run from the repository root:
    python data_analysis/cohort_selection.py

Reads raw DEMO/DPQ and the final feature table; does not modify source data.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nhanes import io as nio  # noqa: E402
from nhanes.paths import WORK  # noqa: E402

CYCLES = nio.CYCLES_2005_2018
SHORT = {cycle: cycle[:4] + '-' + cycle[-2:] for cycle in CYCLES}


def count_by_cycle():
    rows = []
    for cycle in CYCLES:
        demo = nio.read_component('DEMO', cycle, ['RIDAGEYR', 'RIDEXPRG'])
        dpq = nio.read_component('DPQ', cycle, nio.PHQ_ITEMS)
        if demo['SEQN'].duplicated().any() or dpq['SEQN'].duplicated().any():
            raise ValueError(f'Duplicate SEQN in {cycle}')

        age_ok = demo['RIDAGEYR'] >= 20
        items = dpq[nio.PHQ_ITEMS].apply(lambda col: nio.clean_codes(col, [7, 9]))
        # Follow the repository pipeline's actual rule: all nine cleaned values nonmissing.
        complete_phq_ids = set(dpq.loc[items.notna().all(axis=1), 'SEQN'])
        phq_ok = demo['SEQN'].isin(complete_phq_ids)
        not_pregnant = demo['RIDEXPRG'] != 1

        age_n = int(age_ok.sum())
        phq_n = int((age_ok & phq_ok).sum())
        final_n = int((age_ok & phq_ok & not_pregnant).sum())
        rows.append({
            'cycle': SHORT[cycle],
            'All_DEMO': len(demo),
            'Age_20_plus': age_n,
            'Age_20_plus_Valid_PHQ9': phq_n,
            'Final_NonPregnant': final_n,
            'Excluded_Under20_or_AgeMissing': len(demo) - age_n,
            'Excluded_PHQ9_after_age': age_n - phq_n,
            'Excluded_Pregnant_after_PHQ9': phq_n - final_n,
        })
    return pd.DataFrame(rows)


def main():
    summary = count_by_cycle()
    features = pd.read_csv(WORK / 'features' / 'features_2005_2018.csv.gz',
                           usecols=['cycle', 'depression'])
    expected = features.groupby('cycle').size()
    for row in summary.itertuples(index=False):
        actual = int(expected.get(row.cycle, -1))
        if actual != row.Final_NonPregnant:
            raise ValueError(f'{row.cycle}: raw cohort {row.Final_NonPregnant} != feature table {actual}')

    total = {'cycle': 'Overall'}
    for col in summary.columns.drop('cycle'):
        total[col] = int(summary[col].sum())
    result = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)

    outdir = WORK / 'data_analysis'
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / 'cohort_selection.csv'
    result.to_csv(outfile, index=False)
    print(result.to_string(index=False))
    print(f'\nVerified: final cohorts match feature table, N = {len(features):,}')
    print(f'Saved to: {outfile}')
    print('\nNote: counts are unweighted. Pregnancy exclusion exactly follows the')
    print('repository rule RIDEXPRG != 1 (including missing/other codes).')


if __name__ == '__main__':
    main()
