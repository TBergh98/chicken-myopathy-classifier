# AGENTS.md — Agent guidance for this repository

Purpose
- Short, actionable guidance to help AI coding agents be productive in this repo.

Quick commands
- Extract audio features: `python scripts/extract_audio_features.py` (runs pipeline, writes parquet + CSV to `data/processed/audio_features_output`).
- Verify outputs: `python scripts/verify_output.py` (reads parquet/CSV and prints summary).
- Convert session log to Markdown (GUI): `python scripts/session_log_to_markdown.py` (opens file dialog).

Key files & folders
- `data/raw/audios/`: original WAV files (do not modify without user approval).
- `data/processed/audio_features_output/`: pipeline outputs (`audio_features.parquet`, `audio_features.csv`, `extraction_metadata.json`).
- `scripts/`: contains the main runnable scripts: `extract_audio_features.py`, `verify_output.py`, `session_log_to_markdown.py`.
- `docs/`: documentation including `feature_selection.md` and `ml_intern_plan.md`.

Observed dependencies
- numpy, pandas, librosa, soundfile, scipy, tqdm. Use a virtual environment and install relevant packages before running.

Agent guidance / conventions
- Work from the repository root. Scripts resolve paths relative to the workspace root.
- Link, don't embed: prefer referencing docs in `docs/` rather than copying them into instructions.
- Avoid modifying raw audio files; rename operations in the extraction script are deterministic but should be reviewed by a human for collisions.
- Use `verify_output.py` after running the extraction pipeline to sanity-check results.

Conda environment (required)
- **Always** run commands inside the conda environment named `vocalizzazioni-audio`.
- Preferred non-interactive pattern (PowerShell or automation):

```powershell
conda run -n vocalizzazioni-audio --no-capture-output <command>
```

- Interactive / debugging pattern (same shell session):

```powershell
conda activate vocalizzazioni-audio
# then run normal commands, e.g.:
python scripts/extract_audio_features.py
```

- Installing packages inside the env (recommended):

```powershell
conda run -n vocalizzazioni-audio --no-capture-output pip install -r requirements.txt
```

- Create or update the environment from the repository manifest:

```powershell
conda env create -f environment.yml -n vocalizzazioni-audio
conda env update -f environment.yml -n vocalizzazioni-audio
```

- If the agent can access the environment Python binary directly, it may run that interpreter explicitly (non-interactive example):

```powershell
C:\Users\<user>\anaconda3\envs\vocalizzazioni-audio\python.exe scripts/extract_audio_features.py
```

- Rationale: using `conda run` is deterministic and avoids accidental use of the system Python. If the environment does not exist, create it from `environment.yml`.

Where to look first
- Read `scripts/extract_audio_features.py` for the main pipeline and parameters (sample rate, MFCC settings, block size).
- Inspect `docs/feature_selection.md` and `docs/ml_intern_plan.md` for domain-specific decisions.

Suggested next agent customizations
- Add a `requirements.txt` or `.venv` setup instructions linked from this file.
- Optionally add `.github/copilot-instructions.md` for CI or contributor-specific guidance if desired.

If you'd like, I can add a minimal `requirements.txt` and a `.github/copilot-instructions.md` next.
