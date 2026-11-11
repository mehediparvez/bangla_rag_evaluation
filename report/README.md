# Report (LaTeX, Overleaf-ready)

Format follows the undergraduate report your supervisor accepted: title page, bonafide certificate, abstract, acknowledgement, contents/figures/tables lists, five chapters, references.

## Compile
- **Overleaf:** set Menu -> Compiler to **XeLaTeX**. The bibliography uses Biber (Overleaf runs it automatically).
- **Locally:** `latexmk -xelatex main.tex` (needs a TeX distribution with `biber`).

## Fill in
Everything still to do is marked in red as `[TODO: ...]`. Search the files for `\todo` before submitting. Replace `unilogo.png` with your own university logo (the current file is copied from the sample report).

## Results tables
After the runs finish, from the repository root:

```bash
python -m bangla_rag.analyze --results results
python report/make_tables.py --results results     # writes report/tables/*.tex
```

The RQ2 numbers (AUC, false-alarm rate, Spearman) are printed by `python -m bangla_rag.human detect` and `analyze` and are typed into Chapter 4 by hand.
The committed `tables/rq1_*.tex` were generated with `make_tables.latex_table` from the summary printed by `bangla_rag.analyze` in Colab (the result files live on Drive, not in this repo); re-run `make_tables.py` if you have `results/` locally.

## Bibliography
`ref.bib` entries marked `% CHECK` came from search notes. Verify each one on the publisher or arXiv page before submitting.

## Using GitHub with Overleaf
Overleaf's GitHub sync (a paid-plan feature) links **one Overleaf project to the root of one GitHub repository**. Because the report lives in the `report/` subfolder of the code repository, pick one of:

1. **Upload a zip (works on every plan).** Zip the contents of `report/` and use Overleaf -> New Project -> Upload Project. For later changes, re-upload the changed files.
2. **Separate report repository.** Create a new GitHub repo, copy the contents of `report/` into its root, push, and link that repo in Overleaf (Menu -> Sync -> GitHub). Use "Pull GitHub changes" in Overleaf after each push and "Push Overleaf changes to GitHub" for edits made in Overleaf.
