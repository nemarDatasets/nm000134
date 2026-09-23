"""Add a BIDS ``stim_file`` column to every *_events.tsv and describe it in *_events.json.

``stim_file`` names the image shown at each onset, relative to ``stimuli/``:

- ``stim_train`` / ``stim_test``: the image id carried by ``trial_type``
  (``<label>,<image id>,-1,<marker index>``);
- ``oddball``: the target image (16740-16748). In seven sessions every oddball
  marker carries a placeholder id instead (0 or 16539, both training images).
  There the k-th oddball marker takes the k-th oddball of the authors' design
  order (``preprocessed_eeg/sub-*/stim_order.parquet`` in Alljoined/Alljoined-1.6M
  on HuggingFace), a pairing that reproduces the recorded ids in every run
  where both are available;
- ``behav`` and ``debug``: ``n/a`` (a response code and a software marker, not
  image ids).

Usage: python code/add_stim_file.py <dataset root>
"""

import functools
import json
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

TARGETS = set(range(16740, 16749))
EVENTS_JSON = {
    "trial_type": {
        "Description": (
            "Marker written by the presentation software, as "
            "'<label>,<image id>,-1,<marker index>'. label is stim_train or "
            "stim_test (image presentation), oddball (target image of the "
            "attention task) or behav (participant response; the second field is "
            "a response code, which code/generate_behav.py scores as correct for "
            "0 and 3, incorrect for 1 and 2, and no response for 4 and 5); a "
            "few training runs also hold debug markers (second field always 1), "
            "a presentation-software marker that is not a stimulus. "
            "Oddball image ids are placeholders (0 or 16539) in sub-02 ses-02, "
            "sub-04 ses-04, sub-05 ses-04, sub-06 ses-02, sub-06 ses-03, "
            "sub-06 ses-04 and sub-09 ses-02; stim_file holds the image shown."
        )
    },
    "value": {
        "Description": (
            "1-based id of the trial_type string, numbered in order of first "
            "appearance (the row number unless a marker string repeats, as in "
            "sub-20 ses-01); the recordings carry no trigger codes."
        )
    },
    "stim_file": {
        "Description": (
            "Image shown at this onset, relative to stimuli/; n/a for behav and "
            "debug rows. "
            "For oddball rows with a placeholder id it is the target image at the "
            "same position in the authors' design order."
        )
    },
}


@functools.cache
def design_oddballs(subject: str, session: int, block: int) -> list[int]:
    path = f"preprocessed_eeg/sub-{subject}/stim_order.parquet"
    order = pd.read_parquet(
        hf_hub_download("Alljoined/Alljoined-1.6M", path, repo_type="dataset")
    )
    rows = order[
        (order.session == session)
        & (order.block_id == block)
        & (order.partition == "oddball")
    ].sort_values(["sequence_id", "sequence_image_id"])
    return rows.image_path.str[-9:-4].astype(int).tolist()


def stim_files(events: Path, rows: list[list[str]]) -> list[str]:
    markers = [row[2].split(",") for row in rows]
    oddballs = [int(m[1]) for m in markers if m[0] == "oddball"]
    targets = oddballs
    if not set(oddballs) <= TARGETS:
        match = re.search(r"sub-(\d+)_ses-(\d+)_task-images_run-(\d+)", events.name)
        # the design order only covers ses-01..04, not e.g. sub-08 ses-02old
        assert match, f"{events.name}: placeholder oddball ids, no design order"
        subject, session, block = match.groups()
        design = design_oddballs(subject, int(session), int(block))
        assert len(design) >= len(oddballs), (events.name, oddballs, design)
        targets = design[: len(oddballs)]
    files, k = [], 0
    for label, image_id, *_ in markers:
        if label in ("behav", "debug"):
            files.append("n/a")
        elif label == "oddball":
            files.append(f"{targets[k]:05d}.jpg")
            k += 1
        else:
            assert label in ("stim_train", "stim_test"), (events.name, label)
            files.append(f"{int(image_id):05d}.jpg")
    return files


def enrich(events: Path) -> None:
    header, *lines = events.read_text(encoding="utf-8-sig").splitlines()
    assert header == "onset\tduration\ttrial_type\tvalue\tsample", events
    rows = [line.split("\t") for line in lines]
    column = stim_files(events, rows)
    out = [header + "\tstim_file"]
    out += [f"{line}\t{f}" for line, f in zip(lines, column)]
    events.write_text("\ufeff" + "\n".join(out) + "\n", encoding="utf-8")
    sidecar = events.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) | EVENTS_JSON
    sidecar.write_text(json.dumps(meta, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob("sub-*/ses-*/eeg/*_events.tsv"))
    for events in files:
        enrich(events)
    print(f"enriched {len(files)} events files")
