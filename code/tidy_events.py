"""Split the packed marker strings of every *_events.tsv into BIDS columns.

The converter wrote each marker as ``trial_type = '<label>,<value>,-1,<index>'``
(the third field is -1 on every marker). A comma-separated string is fragile
in a TSV cell; mne_bids 0.19, for one, rewrites its '-1,<index>' tail as a
decimal. This script writes:

- ``trial_type``: the label (stim_train, stim_test, oddball, behav, debug);
- ``value``: the marker value as an integer (image id, or response code for
  behav); some sessions zero-padded it, which ``int()`` drops;
- ``marker_index``: the marker's index in the presentation software's stream;
- ``stim_file``: the image shown, relative to ``stimuli/``. In seven sessions
  every oddball marker carries a placeholder value (0 or 16539, both training
  images) instead of a target (16740-16748); there the k-th oddball takes the
  k-th oddball of the authors' design order (``preprocessed_eeg/sub-*/
  stim_order.parquet`` in Alljoined/Alljoined-1.6M on HuggingFace), a pairing
  that reproduces the recorded values in every run where both are available.
  ``n/a`` for behav and debug rows.

Runs on the converter-format files that code/restore_stim_train_events.py
writes. Usage: python code/tidy_events.py <dataset root>
"""

import functools
import json
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

TARGETS = set(range(16740, 16749))
HEADER = "onset\tduration\ttrial_type\tvalue\tsample\tmarker_index\tstim_file"
EVENTS_JSON = {
    "trial_type": {
        "Description": (
            "Marker label written by the presentation software. The original "
            "marker string is '<trial_type>,<value>,-1,<marker_index>', with the "
            "value zero-padded in the sessions listed under value."
        ),
        "Levels": {
            "stim_train": "Training image presentation.",
            "stim_test": "Test image presentation.",
            "oddball": "Target image of the attention task.",
            "behav": "Participant response.",
            "debug": (
                "Presentation-software marker found in a few training runs; "
                "not a stimulus."
            ),
        },
    },
    "value": {
        "Description": (
            "Marker value: the image id for stim_train, stim_test and oddball; "
            "the response code for behav, which code/generate_behav.py scores as "
            "correct for 0 and 3, incorrect for 1 and 2, and no response for 4 "
            "and 5; always 1 for debug. Oddball values are placeholders (0 or "
            "16539) in sub-02 ses-02, sub-04 ses-04, sub-05 ses-04, sub-06 ses-02, "
            "sub-06 ses-03, sub-06 ses-04 and sub-09 ses-02; stim_file holds the "
            "image shown. The original marker strings zero-padded stim_train values "
            "to five digits in sub-07 ses-04, sub-08 ses-02, sub-08 ses-02old, "
            "sub-08 ses-04, sub-11 ses-03, sub-11 ses-04, sub-15 ses-03, "
            "sub-15 ses-04, sub-17 ses-02, sub-17 ses-04 and sub-18 ses-03."
        )
    },
    "marker_index": {
        "Description": (
            "Index of the marker in the presentation software's marker stream "
            "(the last field of the original marker string)."
        )
    },
    "stim_file": {
        "Description": (
            "Image shown at this onset, relative to stimuli/; n/a for behav and "
            "debug rows. For oddball rows with a placeholder value it is the "
            "target image at the same position in the authors' design order."
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


def stim_files(events: Path, markers: list[tuple[str, str, str]]) -> list[str]:
    oddballs = [int(value) for label, value, _ in markers if label == "oddball"]
    targets = oddballs
    if not set(oddballs) <= TARGETS:
        match = re.search(r"sub-(\d+)_ses-(\d+)_task-images_run-(\d+)", events.name)
        # the design order only covers ses-01..04, not e.g. sub-08 ses-02old
        assert match, f"{events.name}: placeholder oddball values, no design order"
        subject, session, block = match.groups()
        design = design_oddballs(subject, int(session), int(block))
        assert len(design) >= len(oddballs), (events.name, oddballs, design)
        targets = design[: len(oddballs)]
    files, k = [], 0
    for label, value, _ in markers:
        if label in ("behav", "debug"):
            files.append("n/a")
        elif label == "oddball":
            files.append(f"{targets[k]:05d}.jpg")
            k += 1
        else:
            assert label in ("stim_train", "stim_test"), (events.name, label)
            files.append(f"{int(value):05d}.jpg")
    return files


def tidy(events: Path) -> None:
    header, *lines = events.read_text(encoding="utf-8-sig").splitlines()
    assert header == "onset\tduration\ttrial_type\tvalue\tsample", events
    rows = [line.split("\t") for line in lines]
    markers = []
    for row in rows:
        label, value, flag, index = row[2].split(",")
        assert flag == "-1" and value.isdigit() and index.isdigit(), (events.name, row[2])
        # int() drops the zero-padding some sessions wrote, so one image has one value
        markers.append((label, str(int(value)), index))
    out = [HEADER]
    for (onset, duration, _, _, sample), (label, value, index), stim_file in zip(
        rows, markers, stim_files(events, markers)
    ):
        out.append(
            f"{onset}\t{duration}\t{label}\t{value}\t{sample}\t{index}\t{stim_file}"
        )
    events.write_text("\n".join(out) + "\n", encoding="utf-8")
    sidecar = events.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) | EVENTS_JSON
    sidecar.write_text(json.dumps(meta, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    files = sorted(Path(sys.argv[1]).glob("sub-*/ses-*/eeg/*_events.tsv"))
    for events in files:
        tidy(events)
    print(f"tidied {len(files)} events files")
