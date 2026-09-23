"""Rebuild nm000134 *_events.tsv rows from the original Alljoined-1.6M EDF annotations.

NEMAR's conversion kept only behav/oddball rows in the training runs (05-19).
The HuggingFace EDFs it was converted from still carry every marker; their
onsets share the BIDS EDFs' time base, so the converter's rows are
``onset = round(hf_onset * 256) / 256``, sorted by onset, with ``value`` the
1-based id of each ``trial_type`` string in order of first appearance.

Usage: python code/restore_stim_train_events.py <nm000134 clone> check|write <run globs...>
"""

import concurrent.futures as cf
import glob
import re
import sys
import tempfile
from pathlib import Path

import mne
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

mne.set_log_level("ERROR")
SFREQ = 256
REPO = "Alljoined/Alljoined-1.6M"
_HF_EDFS: dict[tuple[str, str, str], str] = {}


def hf_index() -> None:
    for f in HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo="raw_eeg", recursive=True):
        m = re.fullmatch(r"raw_eeg/sub-(\d\d)/(session_\d\d(?: old)?)/block_(\d\d)/[^/]+\.edf", f.path)
        if m:
            key = (m[1], m[2], m[3])
            assert key not in _HF_EDFS, key
            _HF_EDFS[key] = f.path


def hf_path(bids_events: Path) -> str:
    m = re.search(r"sub-(\d\d)_ses-(\d\d)(old)?_task-images_run-(\d\d)_events", bids_events.name)
    session = f"session_{m[2]}" + (" old" if m[3] else "")
    return _HF_EDFS[(m[1], session, m[4])]


def render(bids_events: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        edf = hf_hub_download(REPO, hf_path(bids_events), repo_type="dataset", local_dir=tmp)
        raw = mne.io.read_raw_edf(edf)
        assert raw.info["sfreq"] == SFREQ
        onsets, descs = raw.annotations.onset, raw.annotations.description
    samples = np.rint(onsets * SFREQ).astype(int)
    order = np.argsort(samples, kind="stable")
    lines = ["onset\tduration\ttrial_type\tvalue\tsample"]
    ids: dict[str, int] = {}  # repeated marker strings share one id
    for i in order:
        value = ids.setdefault(descs[i], len(ids) + 1)
        lines.append(f"{samples[i] / SFREQ}\t0.0\t{descs[i]}\t{value}\t{samples[i]}")
    return "\ufeff" + "\n".join(lines) + "\n"


def rows(text: str) -> set[tuple[str, str, str]]:
    # (onset, trial_type, sample): `value` is a row number, so it shifts when rows are added
    return {tuple(np.array(line.split("\t"))[[0, 2, 4]]) for line in text.lstrip("\ufeff").splitlines()[1:]}


def process(mode: str, bids_events: Path) -> str:
    old = bids_events.read_text(encoding="utf-8")
    new = render(bids_events)
    if mode == "check":
        return "identical" if new == old else f"DIFF {bids_events.name}"
    # every row NEMAR kept must reappear unchanged
    missing = rows(old) - rows(new)
    if missing:
        return f"ANCHOR MISMATCH {bids_events.name}: {sorted(missing)[:3]}"
    bids_events.write_text(new, encoding="utf-8")
    kinds = [line.split("\t")[2].split(",")[0] for line in new.splitlines()[1:]]
    return f"written stim_train={kinds.count('stim_train')} kept={len(rows(old))}"


if __name__ == "__main__":
    root, mode, patterns = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    files = sorted({Path(p) for pat in patterns for p in glob.glob(str(root / pat))})
    hf_index()
    print(f"{len(files)} files, {len(_HF_EDFS)} HF EDFs indexed", flush=True)
    with cf.ThreadPoolExecutor(8) as pool:
        futures = {pool.submit(process, mode, f): f for f in files}
        for fut in cf.as_completed(futures):
            print(futures[fut].name, fut.result(), flush=True)
