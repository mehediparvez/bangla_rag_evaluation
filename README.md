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
python -m pytest -q                  # 11 unit tests, no key needed
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

```bash
python -m bangla_rag.human sample --runs bm25_c512_k3,hybrid-bge-m3_c512_k3 --n 100
#   grade human/grading_sheet.csv by hand (rubric in human/RUBRIC.txt), without looking at RAGAS scores
pip install -r requirements-ragas.txt
python -m bangla_rag.ragas_eval --tag en                          # judge prompts in English
python -m bangla_rag.ragas_eval --tag bn --prompt-lang bengali    # experimental prompt adaptation
python -m bangla_rag.human analyze --ragas human/ragas_en.csv --ragas human/ragas_bn.csv
```

Reports Spearman rho with bootstrap CIs, the prompt-language difference, and quadratic-weighted kappa if a second rater fills a copy of the sheet.

## Layout

| Path | Purpose |
|---|---|
| `bangla_rag/text.py` | Bangla normalisation and tokenisation (does not split at vowel signs) |
| `bangla_rag/data.py`, `chunking.py` | dataset download/sampling, character chunking |
| `bangla_rag/retrieval.py` | BM25, dense (sentence-transformers + FAISS), hybrid (reciprocal rank fusion) |
| `bangla_rag/prompts.py`, `generation.py` | fixed Bangla prompt, cached rate-limited Gemini client |
| `bangla_rag/run.py`, `analyze.py` | RQ1 experiments, CIs and McNemar tests (Holm-adjusted) |
| `bangla_rag/human.py`, `ragas_eval.py` | RQ2 grading sheet, RAGAS scoring, agreement statistics |
| `tests/` | unit tests |

## Metrics

Exact match and token-F1 against the reference answers; an answer counts as *correct* when F1 >= 0.5 (used for McNemar). Retrieval: passage hit@k (a retrieved chunk comes from the gold passage) and answer-in-context (a reference answer appears in the retrieved text).

## What has and has not been tested

Tested: text/metric/chunking/BM25/fusion/statistics code (unit tests) and a BM25 retrieval-only run on the real dataset.
**Not yet run in the build environment:** dense embedders, FAISS, Gemini generation, RAGAS scoring. Run the Colab notebook's smoke-test cells first and fix any API-version issues before the full run.

## Attribution

BanglaRQA: Ekram et al., 2022, *BanglaRQA: A Benchmark Dataset for Under-resourced Bangla Language Reading Comprehension-based Question Answering with Diverse Question-Answer Types*, EMNLP Findings. Libraries: sentence-transformers, FAISS, rank_bm25, RAGAS (Apache-2.0), google-genai.
