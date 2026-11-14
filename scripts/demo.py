"""A small web interface for the Bangla retrieval-augmented question-answering system (Gradio).

Type a Bangla question, choose a retrieval method, and see the answer together with the passage it was read from.

In Google Colab (private to your notebook session, no public link):
    !pip -q install gradio
    import sys; sys.path.insert(0, "/content/bangla_rag_evaluation/scripts"); sys.path.insert(0, "/content/bangla_rag_evaluation")
    import demo; demo.main(["--model-dir", "/content/drive/MyDrive/bangla_t5_rag/run_fast/best", "--colab-inline"])

On a local machine with a GPU or patience:
    python scripts/demo.py --model-dir <folder with the fine-tuned model>
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_rag.qa_system import METHOD_LABELS, BanglaRAGSystem
from bangla_rag.t5_data import load_items


def pick_examples(data_dir: str, n_answerable: int = 5, n_unanswerable: int = 2, seed: int = 13) -> list[list[str]]:
    """Example questions from the test split (never seen in training); a few are unanswerable on purpose."""
    items = load_items(data_dir, "Test", normalize=lambda s: s)
    rng = random.Random(seed)
    ans = [i["question_raw"] for i in items if i["is_answerable"]]
    unans = [i["question_raw"] for i in items if not i["is_answerable"]]
    return [[q, "bm25"] for q in rng.sample(ans, n_answerable) + rng.sample(unans, n_unanswerable)]


def build_ui(system: BanglaRAGSystem, examples: list[list[str]]):
    import gradio as gr

    labels = {v: k for k, v in METHOD_LABELS.items()}

    def run(question: str, method_label: str):
        question = (question or "").strip()
        if not question:
            return "প্রশ্ন লিখুন (please type a question)", "", [], ""
        res = system.answer(question, labels[method_label], show=3)
        top = res["passages"][0]
        passage = f"{top['title']}  [{top['passage_id']}]\n\n{top['text']}"
        table = [[p["rank"], p["title"], p["passage_id"], round(p["score"], 3)] for p in res["passages"]]
        note = f"{METHOD_LABELS[res['method']]} · {res['seconds']:.2f} s · the model reads only the top-1 passage"
        return res["answer"], passage, table, note

    with gr.Blocks(title="Bangla RAG QA") as ui:
        gr.Markdown("## বাংলা প্রশ্নোত্তর সিস্টেম · Bangla Question Answering with BanglaT5 + retrieval\n"
                    "The system searches 3,000 Bangla Wikipedia passages, gives the best one to a fine-tuned BanglaT5, "
                    "and answers only from that passage. If the passage does not contain the answer it says so.")
        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(label="প্রশ্ন (Question)", lines=2, placeholder="বাংলায় প্রশ্ন লিখুন")
                method = gr.Radio(list(METHOD_LABELS.values()), value=METHOD_LABELS["bm25"], label="Retrieval method")
                button = gr.Button("উত্তর দিন · Answer", variant="primary")
                answer = gr.Textbox(label="উত্তর (Answer)", lines=3)
                info = gr.Markdown()
            with gr.Column(scale=1):
                passage = gr.Textbox(label="Retrieved passage (top-1, read by the model)", lines=12)
                table = gr.Dataframe(headers=["rank", "title", "passage id", "score"], label="Top-3 retrieved", interactive=False)
        gr.Examples(examples=examples, inputs=[question, method], label="Example questions from the test set")
        gr.Markdown("*Examples are test-set questions, including two unanswerable ones. Retrieval can pick the wrong passage; "
                    "when it does, the model usually answers wrongly or gives no answer.*")
        for trigger in (button.click, question.submit):
            trigger(run, [question, method], [answer, passage, table, info])
    return ui


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", default="data/banglarqa")
    ap.add_argument("--cache", default="/content/cache" if Path("/content").exists() else "cache")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--colab-inline", action="store_true", help="show the app inside the Colab cell, without a public link")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link (off by default)")
    args = ap.parse_args(argv)

    system = BanglaRAGSystem.load(args.model_dir, args.data_dir, args.cache, bf16=args.bf16)
    ui = build_ui(system, pick_examples(args.data_dir))
    if args.colab_inline:
        from google.colab import output

        ui.launch(share=False, server_port=args.port, prevent_thread_lock=True, quiet=True)
        output.serve_kernel_port_as_iframe(args.port, height=900)
    else:
        ui.launch(share=args.share, server_port=args.port)
    return ui


if __name__ == "__main__":
    main()
