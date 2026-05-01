"""
Smoke test for nm000134 (Alljoined-1.6M) stimuli alignment.

Walks every ``sub-*/ses-*/eeg/*_events.tsv``, resolves each ``stim_test``
trial via ``StimulusAligner``, and reports per-subject totals plus any
unresolved references.

Run AFTER acquiring stimuli (e.g. from the HuggingFace
``Alljoined/Alljoined-1.6M`` ``stimuli`` subset) into ``stimuli/<id>.jpg``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_stimuli import StimulusAligner

ROOT_DEFAULT = Path(__file__).resolve().parent.parent


def run(root: Path = ROOT_DEFAULT) -> int:
    aligner = StimulusAligner(root)
    tsvs = sorted(root.glob('sub-*/ses-*/eeg/*_events.tsv'))
    if not tsvs:
        print(f'No events.tsv files under {root}')
        return 1
    print(f'== smoke test on {len(tsvs)} events.tsv files ==')

    total_stim = total_resolved = total_missing = 0
    by_subject: dict[str, tuple[int, int, int]] = {}
    missing_ids: set[int] = set()

    for p in tsvs:
        df = pd.read_csv(p, sep='\t')
        paths = aligner.paths_for_events(df)
        n_stim = sum(1 for x in paths if x is not None)
        n_ok = sum(1 for x in paths if x is not None and x.exists())
        n_miss = n_stim - n_ok
        sub = p.parts[-4]
        agg = by_subject.get(sub, (0, 0, 0))
        by_subject[sub] = (agg[0] + n_stim, agg[1] + n_ok, agg[2] + n_miss)
        if n_miss:
            for q in paths:
                if q is not None and not q.exists():
                    try: missing_ids.add(int(q.stem))
                    except ValueError: pass
        total_stim += n_stim
        total_resolved += n_ok
        total_missing += n_miss

    for sub, (s, ok, m) in sorted(by_subject.items()):
        flag = '✓' if m == 0 else '✗'
        print(f'  {flag} {sub}: stim_test={s:8d}  resolved={ok:8d}  missing={m:6d}')

    print()
    print('== summary ==')
    print(f'   subjects/runs   : {len(tsvs)} events.tsv files, {len(by_subject)} subjects')
    print(f'   total stim_test : {total_stim}')
    print(f'   resolved+exists : {total_resolved}')
    print(f'   resolved+missing: {total_missing}')
    if missing_ids:
        print(f'   distinct missing image_ids: {len(missing_ids)} (first 10: {sorted(missing_ids)[:10]})')
    return 1 if total_missing else 0


if __name__ == '__main__':
    sys.exit(run())
