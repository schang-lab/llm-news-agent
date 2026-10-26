# Whose Story Does AI Tell?

This repository includes the data, code, and experimental results outlined in our paper, "Whose Story Does AI Tell?". In these files, we create an LLM-based news agent that responds to a user-query by retrieving and synthesizing informaiton a corpus of news articles sourced from AllSides. We study how this agent mediates the information that is being raised to the user by analyzing the distribution of frames in retrieved articles and generated responses compared to the input article set. Note that our data provided is incomplete: in order to respect the authorship of the articles used in this experiment, we do not provide the body of the article text or the generations which may quote the articles, all marked as [REDACTED]. We do, however, provide metadata to enable future researchers to reconstruct our dataset.

The `data_public/` folder contains the CSVs from our work.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python code is in `scripts/`, with shared helpers in `scripts/support/`.
Each script takes input paths and `-o` for the output CSV. You can choose a
different output path. `--output` also works. Inputs and outputs may be `.csv` or `.csv.gz`;
gzip compression is handled automatically.

## Framing

Extract article frames, group them, and assign each extraction to a group:

```bash
python3 scripts/extract_frames.py --corpus path/to/corpus -o path/to/extracted/frames

python3 scripts/group_frames.py --extracted-frames path/to/extracted/frames \
  -o path/to/grouped/frames

python3 scripts/assign_frames.py --extracted-frames path/to/extracted/frames \
  --groups path/to/grouped/frames \
  -o path/to/assigned/frames

python3 scripts/count_frames.py --groups  path/to/grouped/frames \
  --assignments path/to/assigned/frames \
  -o path/to/grouped/frames
```

Human review produces the final list of frames (example at `data_public/frames_public.csv`), which is already supplied.
Use it to score every article:

```bash
python3 scripts/annotate_corpus.py --corpus path/to/corpus --frames path/to/frames \
  -o path/to/annotations
```

## Retrieval

```bash
python3 scripts/setup_retrieval.py --corpus path/to/corpus --queries path/to/queries \
  -o path/to/retrieval/experiments

python3 scripts/execute_retrieval.py --corpus path/to/corpus \
  --experiments path/to/retrieval/experiments -o path/to/retrieval/results
```

## Synthesis

Each answer uses the same model and ordered article set as its LLM retrieval.
Then score the answer against the event's frames:

```bash
python3 scripts/setup_synthesis.py --corpus path/to/corpus \
  --retrieval-results path/to/retrieval/results -o path/to/synthesis/experiments

python3 scripts/execute_synthesis.py --experiments path/to/synthesis/experiments \
  -o path/to/synthesis/results

python3 scripts/annotate_synthesis.py --results path/to/synthesis/results \
  --frames path/to/frames -o path/to/annotated/synthesis/results
```

## Analysis

Analysis and figure generation is in `visualizations.ipynb`.

## Notes

The corpus contains 5,061 articles across 26 events and 117 final frames.
Models, settings, and seeds are in `scripts/config.py`.
