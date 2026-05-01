"""
Align nm000134 (Alljoined-1.6M) events.tsv rows with their stimulus images.

events.tsv ``trial_type`` values come in three flavours:
    stim_test,<image_id>,-1,<seq>   - image presentation event
    behav,<v>,-1,<n>                - catch / oddball behavioural event
    oddball                          - oddball marker

After acquiring the stimuli (e.g. via ``huggingface_hub.snapshot_download
(repo_id='Alljoined/Alljoined-1.6M', repo_type='dataset',
allow_patterns='stimuli/**')`` or this dataset's S3 bucket once published),
the ``stimuli/`` folder contains ``<image_id>.jpg`` files matching the
numeric ids in the ``stim_test`` trial type.

Usage
-----
    aligner = StimulusAligner(root='/path/to/nm000134')
    paths = aligner.paths_for_events(events_df)         # list[Path | None]
    img   = aligner.image_for_event(row, mode='PIL')    # single row
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from PIL import Image as _PIL_Image
except ImportError:
    _PIL_Image = None


def parse_trial_type(s: object) -> Optional[int]:
    """Return the image_id for a stim_test trial, or None for behav/oddball/empty."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    parts = str(s).split(',', 3)
    if len(parts) < 2 or parts[0] != 'stim_test':
        return None
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return None


class StimulusAligner:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.stim_root = self.root / 'stimuli'

    # Filename pattern in the upstream HF dataset: 5-digit zero-padded
    # (e.g. image id 16641 -> `16641.jpg`, image id 99 -> `00099.jpg`).
    FILENAME_TEMPLATE = '{:05d}.jpg'

    def path_for_event(self, row) -> Optional[Path]:
        ttype = row.get('trial_type') if hasattr(row, 'get') else getattr(row, 'trial_type', None)
        image_id = parse_trial_type(ttype)
        if image_id is None:
            return None
        return self.stim_root / self.FILENAME_TEMPLATE.format(image_id)

    def image_for_event(self, row, mode: str = 'PIL'):
        p = self.path_for_event(row)
        if p is None:
            return None
        if mode == 'path':
            return p
        if mode == 'bytes':
            return p.read_bytes()
        if _PIL_Image is None:
            raise RuntimeError("Pillow is not installed; use mode='path' or 'bytes'.")
        return _PIL_Image.open(p)

    def paths_for_events(self, events: pd.DataFrame) -> list[Optional[Path]]:
        if 'trial_type' not in events.columns:
            return [None] * len(events)
        ids = events['trial_type'].map(parse_trial_type)
        return [
            None if pd.isna(x) else self.stim_root / self.FILENAME_TEMPLATE.format(int(x))
            for x in ids
        ]


def demo(
    root: str = '/data/tau/iceberg_1/titanic_1/datasets/bids/nm000134',
    subject: str = '01',
    session: str = '01',
    run: str = '01',
) -> None:
    """Resolve a sample events.tsv and report counts."""
    root_p = Path(root)
    aligner = StimulusAligner(root_p)
    ev = root_p / f'sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-images_run-{run}_events.tsv'
    df = pd.read_csv(ev, sep='\t')
    paths = aligner.paths_for_events(df)
    n_stim = sum(1 for p in paths if p is not None)
    n_other = sum(1 for p in paths if p is None)
    n_exists = sum(1 for p in paths if p is not None and p.exists())
    print(f'== sub-{subject} ses-{session} run-{run} ==')
    print(f'   total rows: {len(df)}')
    print(f'   stim_test rows: {n_stim} (resolved+exists: {n_exists})')
    print(f'   non-stim rows: {n_other}')


if __name__ == '__main__':
    demo()
