# Report (LaTeX, Overleaf-ready)

Title page, bonafide certificate, abstract, acknowledgement, contents/figures/tables lists, five chapters, references.

## Compile
- **Overleaf:** the default compiler works; the bibliography uses Biber, which Overleaf runs automatically. If Bangla script is added later (for example example questions), switch Menu -> Compiler to **XeLaTeX** and set a Bangla font.
- **Locally:** `latexmk -xelatex main.tex` (needs a TeX distribution with `biber`).

## To fill in
Everything still to do is marked in red as `[TODO: ...]`. Search the files for `\todo` before submitting.
`unilogo.png` is the university logo.

## Results
Training log, loss curve and per-question predictions come from the Colab run (Google Drive folder `bangla_t5_rag`). Copy `training_curve.png` into `figures/`.

## Bibliography
`ref.bib` entries marked `% CHECK` were written from memory or search notes. Verify each one on the publisher, ACL Anthology or arXiv page before submitting.

## Overleaf
GitHub sync is a paid feature. On the free plan, zip the contents of `report/` and use New Project -> Upload Project.
