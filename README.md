# Team ChivaCF — Football Possession Segmentation (More Than A Hack 2027)

Deterministic spatiotemporal pipeline to generate frame-level possession labels (`H`, `A`, `D`, `X`) across all `199,319` frames of the tactical match feed.

## Repository Structure
* `submission_generator.py`: Main pipeline script (spatial proximity control, 125-frame temporal mode smoothing, bidirectional gap bridging, ball-occlusion abstention guard, micro-possession pruning, and `SET PIECES` dead-ball injection).
* `games.py`: Official competition asset path resolver.
* `requirements.txt`: Pinned Python dependencies.
* `evaluation/validate_submission.py`: Official format and schema validator.

## Installation
1. Install Python 3.10+ and the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the official competition `data/` folder in the repository root (or set the `MTAH_DATA_DIR` environment variable). *Match data is excluded from this repository per competition rules.*

## Reproduce `chivacf.csv`
Run the pipeline from the repository root:
```bash
python submission_generator.py
```
The output file will be generated at `evaluation/results/chivacf.csv`.

## Validate Format
Verify that the generated file passes all official format checks:
```bash
python evaluation/validate_submission.py evaluation/results/chivacf.csv
```