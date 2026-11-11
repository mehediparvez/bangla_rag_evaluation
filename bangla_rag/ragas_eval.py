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


# Hand-written Bengali versions of the RAGAS instructions (keys are prompt class names). Used only when RAGAS'
# automatic translation of the few-shot examples fails; the examples then stay in English ("bengali-instruction").
# Digits are kept ASCII (1 / 0) because the verdict fields are parsed as integers.
MANUAL_BN_INSTRUCTIONS = {
    "StatementGeneratorPrompt": (
        "একটি প্রশ্ন এবং একটি উত্তর দেওয়া হলো। উত্তরের প্রতিটি বাক্যের জটিলতা বিশ্লেষণ করো। "
        "প্রতিটি বাক্যকে এক বা একাধিক সম্পূর্ণ বোধগম্য বিবৃতিতে ভেঙে লেখো। "
        "কোনো বিবৃতিতে সর্বনাম ব্যবহার করবে না। আউটপুট JSON ফরম্যাটে দাও।"
    ),
    "NLIStatementPrompt": (
        "তোমার কাজ হলো প্রদত্ত প্রসঙ্গের (context) ভিত্তিতে কয়েকটি বিবৃতি কতটা বিশ্বস্ত (faithful) তা যাচাই করা। "
        "প্রতিটি বিবৃতির জন্য verdict হিসেবে 1 দাও যদি বিবৃতিটি সরাসরি প্রসঙ্গ থেকে অনুমান করা যায়, "
        "অথবা 0 দাও যদি সরাসরি অনুমান করা না যায়।"
    ),
    "ResponseRelevancePrompt": (
        "প্রদত্ত উত্তরের জন্য একটি প্রশ্ন তৈরি করো এবং উত্তরটি অস্পষ্ট (noncommittal) কি না চিহ্নিত করো। "
        "উত্তর অস্পষ্ট হলে noncommittal এর মান 1 এবং স্পষ্ট হলে 0 দাও। "
        "অস্পষ্ট উত্তর হলো যা এড়িয়ে যায়, অনিশ্চিত বা দ্ব্যর্থক; যেমন \"আমি জানি না\" বা \"আমি নিশ্চিত নই\"।"
    ),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default="human/grading_sheet.csv")
    ap.add_argument("--tag", default="en", help="label for this configuration, used in the output file name")
    ap.add_argument("--judge", default="gemini-3.5-flash-lite")
    ap.add_argument("--embedding-model", default="models/gemini-embedding-001")
    ap.add_argument("--prompt-lang", default=None, help="adapt RAGAS prompts to this language, e.g. 'bengali'")
    ap.add_argument("--all-metrics", action="store_true", help="also compute context precision and recall")
    ap.add_argument("--workers", type=int, default=1, help="keep low on the Gemini free tier")
    ap.add_argument("--rpm", type=float, default=12, help="max judge requests per minute (free tier is 15)")
    ap.add_argument("--chunk", type=int, default=10, help="rows scored per batch; progress is saved after each batch")
    ap.add_argument("--strictness", type=int, default=3,
                    help="ResponseRelevancy questions generated per answer (RAGAS default 3; 1 uses ~40%% fewer calls)")
    ap.add_argument("--reverse", action="store_true",
                    help="score the last rows first (the newest, unsupported-rich rows), so a quota stop still leaves a useful subset")
    ap.add_argument("--out-dir", default="human")
    args = ap.parse_args(argv)

    from langchain_core.rate_limiters import InMemoryRateLimiter
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
    limiter = InMemoryRateLimiter(requests_per_second=args.rpm / 60.0, max_bucket_size=1)
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=args.judge, temperature=0, google_api_key=key,
                                                    rate_limiter=limiter))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=args.embedding_model, google_api_key=key))

    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb, strictness=args.strictness)]
    if args.all_metrics:
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
        metrics += [LLMContextPrecisionWithReference(llm=llm), LLMContextRecall(llm=llm)]

    if args.prompt_lang:  # experimental: translate RAGAS' internal prompts
        # adapt_prompts() only RETURNS the translated prompts; they must be applied with set_prompts().
        # The translation is an LLM call that sometimes fails, so retry, and cache the result on disk.
        prompt_dir = Path(args.out_dir) / f"prompts_{args.prompt_lang}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for m in metrics:
            if not hasattr(m, "adapt_prompts"):
                continue
            loaded = None
            for lang in (args.prompt_lang, f"{args.prompt_lang}-instruction"):  # full translation, or the fallback variant
                try:
                    m.set_prompts(**m.load_prompts(str(prompt_dir), lang))
                    loaded = lang
                    break
                except Exception:  # noqa: BLE001
                    continue
            if loaded:
                print(f"{m.name}: loaded saved '{loaded}' prompts from {prompt_dir}")
            else:
                for attempt in range(1, 4):
                    try:
                        m.set_prompts(**asyncio.run(m.adapt_prompts(language=args.prompt_lang, llm=llm)))
                        m.save_prompts(str(prompt_dir))
                        print(f"{m.name}: adapted prompts to '{args.prompt_lang}' (attempt {attempt}), saved to {prompt_dir}")
                        break
                    except Exception as e:  # noqa: BLE001
                        print(f"{m.name}: prompt adaptation attempt {attempt} failed: {str(e)[:120]}")
                else:
                    if args.prompt_lang != "bengali":
                        raise SystemExit(f"could not adapt {m.name} prompts to {args.prompt_lang}; try again later")
                    import copy

                    fallback = {}
                    for pname, prompt in m.get_prompts().items():
                        text = MANUAL_BN_INSTRUCTIONS.get(type(prompt).__name__)
                        if text:
                            q = copy.deepcopy(prompt)
                            q.instruction, q.language = text, "bengali-instruction"
                            fallback[pname] = q
                    m.set_prompts(**fallback)
                    m.save_prompts(str(prompt_dir))
                    print(f"{m.name}: automatic translation failed; using hand-written Bengali instructions "
                          f"(few-shot examples stay English). Report this variant as 'Bengali instructions only'.")
            print(f"  {m.name} prompt languages in use: { {n: p.language for n, p in m.get_prompts().items()} }")

    keep_cols = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.out_dir) / f"ragas_{args.tag}.csv"

    # Resume: rows already scored (non-NaN faithfulness) in an earlier run are kept.
    done = pd.DataFrame(columns=["row_id"])
    if path.exists():
        done = pd.read_csv(path, encoding="utf-8-sig")
    scored_ids = set(done.loc[done.get("faithfulness").notna(), "row_id"]) if "faithfulness" in done else set()
    pending = [i for i in range(len(samples)) if int(sheet["row_id"].iloc[i]) not in scored_ids]
    if args.reverse:
        pending.reverse()
    print(f"{len(scored_ids)} rows already scored, {len(pending)} to do (saving every {args.chunk} rows)")

    for c0 in range(0, len(pending), args.chunk):
        idx = pending[c0:c0 + args.chunk]
        result = evaluate(EvaluationDataset.from_list([samples[i] for i in idx]), metrics=metrics, llm=llm,
                          embeddings=emb, run_config=RunConfig(max_workers=args.workers, timeout=180, max_retries=6))
        scores = result.to_pandas()
        keep = [c for c in scores.columns if c in keep_cols]
        part = pd.concat([sheet[["row_id"]].iloc[idx].reset_index(drop=True), scores[keep].reset_index(drop=True)], axis=1)
        if "faithfulness" in part and part["faithfulness"].isna().all():
            print("\nEvery call in this batch failed (most likely the daily quota). Progress so far is saved; "
                  "re-run the same command later or with another API key.")
            break
        done = pd.concat([done[~done["row_id"].isin(part["row_id"])], part], ignore_index=True)
        done.sort_values("row_id").to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved {len(done)} rows to {path}")
    print(f"finished: {len(done)}/{len(samples)} rows in {path}; NaN means the judge call failed for that row")


if __name__ == "__main__":
    main()
