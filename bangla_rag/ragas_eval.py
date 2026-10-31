"""Score the human-graded answers with RAGAS (RQ2).

Written against the ragas 0.2.x API (``requirements-ragas.txt`` pins it). RAGAS
changes quickly between releases; if a import below fails, check the installed
version first. The judge is Gemini through LangChain.

    python -m bangla_rag.ragas_eval --sheet human/grading_sheet.csv --tag en
    python -m bangla_rag.ragas_eval --sheet human/grading_sheet.csv --tag bn --prompt-lang bengali
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default="human/grading_sheet.csv")
    ap.add_argument("--tag", default="en", help="label for this configuration, used in the output file name")
    ap.add_argument("--judge", default="gemini-2.5-flash-lite")
    ap.add_argument("--embedding-model", default="models/gemini-embedding-001")
    ap.add_argument("--prompt-lang", default=None, help="adapt RAGAS prompts to this language, e.g. 'bengali'")
    ap.add_argument("--all-metrics", action="store_true", help="also compute context precision and recall")
    ap.add_argument("--workers", type=int, default=2, help="keep low on the Gemini free tier")
    ap.add_argument("--out-dir", default="human")
    args = ap.parse_args(argv)

    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY first.")

    sheet = pd.read_csv(args.sheet, encoding="utf-8-sig")
    samples = [
        {"user_input": r.question, "retrieved_contexts": str(r.contexts).split("\n---\n"),
         "response": str(r.response), "reference": str(r.reference)}
        for r in sheet.itertuples()
    ]
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=args.judge, temperature=0, google_api_key=key))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=args.embedding_model, google_api_key=key))

    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb)]
    if args.all_metrics:
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
        metrics += [LLMContextPrecisionWithReference(llm=llm), LLMContextRecall(llm=llm)]

    if args.prompt_lang:  # experimental: translate RAGAS' internal prompts
        for m in metrics:
            if hasattr(m, "adapt_prompts"):
                asyncio.run(m.adapt_prompts(language=args.prompt_lang, llm=llm))

    result = evaluate(EvaluationDataset.from_list(samples), metrics=metrics, llm=llm, embeddings=emb,
                      run_config=RunConfig(max_workers=args.workers, timeout=180, max_retries=6))
    scores = result.to_pandas()
    keep = [c for c in scores.columns if c in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall")]
    out = pd.concat([sheet[["row_id"]].reset_index(drop=True), scores[keep].reset_index(drop=True)], axis=1)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.out_dir) / f"ragas_{args.tag}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved {len(out)} rows to {path}; NaN means the judge call failed for that row")


if __name__ == "__main__":
    main()
