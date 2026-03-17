[![DOI](https://img.shields.io/badge/DOI-10.82901%2Fnemar.nm000134-blue)](https://doi.org/10.82901/nemar.nm000134)

# Alljoined-1.6M: Million-Trial EEG Dataset with Consumer-Grade Hardware

## Overview

Alljoined-1.6M is a large-scale EEG dataset of neural responses to rapid serial visual presentation (RSVP) of natural images, recorded using a consumer-grade 32-channel EMOTIV FLEX2 system. Twenty healthy adult participants (ages 23-63; 15 male, 5 female) each completed four recording sessions, generating over 1.6 million visual stimulus trials in total.

The dataset was designed to evaluate whether deep neural network-based brain-computer interface (BCI) research and semantic decoding methods can be effectively conducted with affordable consumer-grade EEG systems (approximately $2.2k versus $35-60k for research-grade systems).

**Reference:** Xu, J., Bruzadin Nunes, U., Jiang, W., Ryther, S., Pringle, J., Scotti, P. S., Delorme, A., & Kneeland, R. (2025). Alljoined-1.6M: A Million-Trial EEG-Image Dataset for Evaluating Affordable Brain-Computer Interfaces. <https://doi.org/10.48550/arXiv.2508.18571>

## Recording Setup

- **Equipment:** EMOTIV FLEX2, 32-channel sintered Ag/AgCl gel-based electrodes
- **Connectivity:** wireless Bluetooth 5.2
- **Sampling rate:** 256 Hz (resampled to 250 Hz in published analyses)
- **Montage:** extended 10-20 system, focused on occipital/visual regions
- **Channels:** Cz, Fp1, F7, F3, CP5, CP1, P1, P3, P5, P7, PO9, PO7, PO3, O1, O9, Pz, POz, Oz, O10, O2, PO4, PO8, PO10, P8, P6, P4, P2, CP2, CP6, F4, F8, Fp2
- **Firmware filters:** dual 50/60 Hz notch filter (built into EMOTIV firmware)
- **Cost:** approximately $2.2k (approximately 27x cheaper than research-grade systems)

## Task Paradigm

Rapid Serial Visual Presentation (RSVP) with orthogonal oddball detection. Each trial consisted of an image presented for 100 ms, followed by 100 ms of blank screen (200 ms total cycle). A small semi-transparent red fixation dot (0.2 x 0.2 degrees, 50% opacity) was present throughout.

Oddball detection: participants pressed a button when they detected catch trials featuring a Woody (Toy Story) character, which appeared in approximately 6% of sequences. Detection window was up to 2 seconds post-sequence. This task maintained engagement without biasing perception toward specific image categories.

Viewing distance: 60 cm; viewing angle: 7 degrees.

## Stimulus Set

16,740 unique images from the THINGS database (26,000 total images across 1,854 object categories), identical to the THINGS-EEG2 stimulus set for direct comparison.

- **Test images:** shown 80 times per participant (4 sessions x 4 test blocks x 5 presentations)
- **Training images:** shown 4-5 times per participant
- **Randomization:** constrained so no image repeats within 2 intervening items

## Subjects, Sessions, and Runs

20 subjects, 4 sessions each (sub-08 has an additional session `ses-02old`, a retake of session 2). Each session contains 19 RSVP blocks (runs), approximately 5 minutes each. The first 4 runs per session present test images; the remaining 15 runs present training images.

Total: 83,520 image trials per subject; approximately 1.6 million trials across all 20 participants.

| Subject | Sessions | Runs | Notes |
|---------|----------|------|-------|
| sub-01 | 4 | 76 | |
| sub-02 | 4 | 76 | |
| sub-03 | 4 | 76 | |
| sub-04 | 4 | 76 | |
| sub-05 | 4 | 76 | |
| sub-06 | 4 | 76 | |
| sub-07 | 4 | 76 | |
| sub-08 | 5 | 81 | Includes ses-02old (session 2 retake) |
| sub-09 | 4 | 76 | |
| sub-10 | 4 | 76 | |
| sub-11 | 4 | 76 | |
| sub-12 | 4 | 76 | |
| sub-13 | 4 | 76 | |
| sub-14 | 4 | 76 | |
| sub-15 | 4 | 76 | |
| sub-16 | 4 | 76 | |
| sub-17 | 4 | 76 | |
| sub-18 | 4 | 76 | |
| sub-19 | 4 | 76 | |
| sub-20 | 4 | 76 | |

Participants were recruited from San Francisco via local platforms (Craigslist 55%, Instawork 35%) and filtered from an initial pool of 48 for high behavioral engagement. Mean oddball detection performance: 88% AUC (+/- 1% SE).

## Data Format

Raw continuous EEG recordings are stored as European Data Format (EDF) files, the native export format of the EMOTIV FLEX2 system (16-bit resolution). Only the 32 EEG channels are retained; EMOTIV metadata channels (timestamps, counters, contact quality, motion sensors, etc.) were excluded during conversion.

**Per-run files:**

| Path | Description |
|------|-------------|
| `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-images_run-ZZ_eeg.edf` | Raw EEG |
| `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-images_run-ZZ_events.tsv` | Events |
| `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-images_run-ZZ_events.json` | Event metadata |
| `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-images_run-ZZ_channels.tsv` | Channels |
| `sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-images_run-ZZ_eeg.json` | Recording parameters |
| `sub-XX/ses-YY/eeg/sub-XX_ses-YY_space-CapTrak_coordsystem.json` | Coordinate system |
| `sub-XX/ses-YY/eeg/sub-XX_ses-YY_space-CapTrak_electrodes.tsv` | Electrode positions |

Event annotations in the events.tsv files use the following `trial_type` format from the EMOTIV recording system:

- `stim_test,{image_id},-1,{trial}` -- test image presentation
- `oddball,...` -- oddball (catch) trial
- `behav,...` -- behavioral response (button press)

## Source Data

The `sourcedata/` directory contains the original EMOTIV JSON metadata files from each recording block. These files include the raw EMOTIV marker data with precise timestamps, UUIDs, and port information as recorded by the EMOTIV software. They are the original, unprocessed recording artifacts from the EMOTIV system, not derived products, and are stored in `sourcedata/` per BIDS conventions.

```
sourcedata/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-images_run-ZZ_recording.json
```

## Code

The `code/` directory contains the original Alljoined-1.6M analysis code, cloned from <https://github.com/Alljoined/Alljoined-1.6M>.

## BIDS Conversion

Converted to BIDS by Yahya Shirazi (Swartz Center for Computational Neuroscience, UC San Diego) using MNE-BIDS and custom scripts.

- **Source data:** HuggingFace <https://huggingface.co/datasets/Alljoined/Alljoined-1.6M>
- EMOTIV channel `Afz` renamed to `AFz` (standard 10-20 capitalization)
- Session label `session_02 old` sanitized to `ses-02old` for BIDS compliance
- 95 EMOTIV metadata channels excluded (only 32 EEG channels retained)
- Conversion validated with round-trip integrity checks (data amplitude, per-channel correlation, sampling frequency, event count, and event timing)

## License and Terms of Use

This dataset is distributed under CC-BY-NC-ND-4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0) with the following additional terms imposed by the Alljoined team. By using this dataset you agree to all conditions below.

1. Researcher shall use the Dataset only for non-commercial research and educational purposes, in accordance with Alljoined's [Terms of Use](https://www.alljoined.com/terms-of-use).
2. **No Warranties:** Alljoined makes no representations or warranties regarding the Dataset, including but not limited to warranties of non-infringement or fitness for a particular purpose.
3. **Full Responsibility:** Researcher accepts full responsibility for his or her use of the Dataset and shall defend and indemnify Alljoined, including their employees, officers and agents, against any and all claims arising from Researcher's use of the Dataset.
4. **Privacy Compliance:** Researcher shall comply with Alljoined's [Privacy Policy](https://www.alljoined.com/privacy-policy) and ensure that any use of the Dataset respects the privacy rights of individuals whose data may be included.
5. **Sharing Rights:** Researcher may provide research associates and colleagues with access to the Dataset provided that they first agree to be bound by these terms and conditions.
6. **Termination Rights:** Alljoined reserves the right to terminate Researcher's access to the Dataset at any time.
7. **Commercial Entity Binding:** If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.
8. **Governing Law:** The law of the State of California shall apply to all disputes under this agreement.

- Full terms: <https://www.alljoined.com/terms-of-use>
- Privacy policy: <https://www.alljoined.com/privacy-policy>

## References

Xu, J., Bruzadin Nunes, U., Jiang, W., Ryther, S., Pringle, J., Scotti, P. S., Delorme, A., & Kneeland, R. (2025). Alljoined-1.6M: A Million-Trial EEG-Image Dataset for Evaluating Affordable Brain-Computer Interfaces. https://doi.org/10.48550/arXiv.2508.18571

Xu, J., Lee, S. K., & Jiang, W. (2024). Alljoined -- A dataset for EEG-to-Image decoding. https://doi.org/10.48550/arXiv.2404.05553
