# What makes Bangla RAG answers trustworthy?

MSc project code. Two questions:

- **RQ1** Which embedder, chunk size and retrieval method (BM25 / dense / hybrid) give the most accurate Bangla RAG answers?
- **RQ2** Do automatic RAGAS judge scores agree with human grades on Bangla answers?

Data: [BanglaRQA](https://huggingface.co/datasets/sartajekram/BanglaRQA) (Ekram et al., EMNLP Findings 2022; CC BY-NC-SA 4.0), general-domain Bangla Wikipedia passages with questions. All 3,000 passages form the retrieval pool; questions come from a seeded sample of answerable *test* questions. The dataset is downloaded at run time and is not stored in this repository.
Generator: Gemini 3.5 Flash-Lite (override with `--model`), temperature 0, one fixed prompt for every run.

> Domain note: the original plan used medical biology (BanglaMedQA) with a textbook corpus. The textbook text was not obtainable in time, so this version uses BanglaRQA. State this as a limitation in the thesis.

## Quick start (Colab or local)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # Colab: use Secrets, see notebooks/colab.ipynb
python -m pytest -q                  # 19 unit tests, no key needed
python -m bangla_rag.data            # download BanglaRQA

# 1. retrieval metrics only: free, no key, uses the GPU for dense models
python -m bangla_rag.run --preset core --retrieval-only --n 0

# 2. full RQ1 run with Gemini (resumable; free tier needs --rpm 10 or lower)
python -m bangla_rag.run --preset core --n 200 --rpm 10
python -m bangla_rag.analyze          # -> results/summary.md
```

Presets: `core` (no retrieval, BM25, dense bge-m3, hybrid) and `full` (adds Qwen3-Embedding, multilingual-e5, chunk 256/1000, top-5). Or pick runs by id: `--runs none,bm25_c512_k3`.
Default configuration: dense bge-m3, 512-character chunks, top-3. Each other run changes one factor from it.

The Gemini free tier has daily request limits. If it is hit, the script saves progress and exits with code 2; run the same command again later and it continues (every call is cached in `results/gemini_cache.jsonl`).

## RQ2

RAGAS faithfulness and answer relevance are compared with reference grades on the same answers. Faithful answers dominate in a
random sample (almost all grades are 3), so a plain correlation has no variance to work with. The design therefore adds
known-unsupported answers and reports *detection* (AUC, false-alarm rate) next to Spearman rho:

| Row type | How it is made | Flag |
|---|---|---|
| random | answers from retrieval runs | `--n` |
| hard | gold answer not in context, or F1 < 0.5 | `--hard-frac` |
| closed-book | no-retrieval answer shown with retrieved passages | `--closed-book N` |
| synthetic | answer paired with another question's passages (ids in `human/perturb_key.csv`) | `--perturb N` |

```bash
python -m bangla_rag.human sample --runs bm25_c512_k3 --n 0 --closed-book 60 --perturb 30        --extend human/grading_sheet_v2.csv --out human/grading_sheet_v3.csv
#   grade the new rows by hand (rubric: human/RUBRIC.txt), without looking at RAGAS scores
pip install -r requirements-ragas.txt
python -m bangla_rag.ragas_eval --sheet human/grading_sheet_v3.csv --tag en                         # English judge prompts
python -m bangla_rag.ragas_eval --sheet human/grading_sheet_v3.csv --tag bn --prompt-lang bengali   # Bengali prompts
python -m bangla_rag.human detect  --sheet human/grading_sheet_v3.csv --ragas human/ragas_en.csv --ragas human/ragas_bn.csv
python -m bangla_rag.human analyze --sheet human/grading_sheet_v3.csv --ragas human/ragas_en.csv --ragas human/ragas_bn.csv
```

`--rater ai` makes `detect` and `analyze` read the `ai_faithful` / `ai_relevant` columns instead of `human_*`.
**Any AI-assisted grade must be disclosed** as such in the thesis, and agreement between a rater and labels they adopted from
the AI is 1.0 by construction, so it must not be reported as inter-rater agreement.
`ragas_eval` rate-limits judge calls, saves every batch and resumes, so a quota stop or Colab disconnect loses at most one batch.

## Metrics

Exact match and token-F1 against the reference answers; an answer counts as *correct* when F1 >= 0.5 (used for McNemar). Retrieval: passage hit@k (a retrieved chunk comes from the gold passage) and answer-in-context (a reference answer appears in the retrieved text).

## What has and has not been tested

Unit tests (19, no API key or network): text/metric/chunking/BM25/fusion/statistics code and the RQ2 sampling and analysis code. The full pipeline (dense retrieval, Gemini generation, RAGAS scoring) is exercised end to end in the Colab notebook; its results are in `results/summary.md`.

## Attribution

BanglaRQA: Ekram et al., 2022, *BanglaRQA: A Benchmark Dataset for Under-resourced Bangla Language Reading Comprehension-based Question Answering with Diverse Question-Answer Types*, EMNLP Findings. Libraries: sentence-transformers, FAISS, rank_bm25, RAGAS (Apache-2.0), google-genai.
