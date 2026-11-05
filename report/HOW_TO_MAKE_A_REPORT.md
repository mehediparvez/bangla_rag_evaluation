# How to make a project report in Overleaf with Claude Code

A reusable recipe for turning a finished (or running) project into a LaTeX report in a format your supervisor already accepts. It works for any project, not only this one.

## What you need
- A sample report your supervisor has accepted before (the `.tex` file, `.bib` file and images).
- Your project folder, ideally a git repository, with the code and results.
- Claude Code open in the project folder.
- An Overleaf account (free is enough).

## Step 1: Put the sample next to your project
Copy the old report into a folder such as `sample_report/` inside your project. Claude reads it to copy the layout: page size, margins, line spacing, front pages, chapter order and bibliography style.

## Step 2: Ask Claude to build the skeleton
Paste a prompt like this and fill in the brackets:

> Read `sample_report/main.tex`. Create a report for my project in a new `report/` folder, using the same format (title page, certificate, abstract, acknowledgement, contents, chapters, references). Keep my university, department, supervisor and ID details exactly as in the sample. Write the chapters from my code, README and notes. Do not invent any numbers: where results are not final, put a visible `\todo{...}` placeholder. Add only references you can support, and mark any unverified ones with a `% CHECK` comment.

Ask for a plan first if you want to review the structure before files are written.

## Step 3: Check what Claude produced
- Search every file for `\todo` and fill each one in or delete it.
- Read the abstract, introduction and limitations yourself. You must be able to defend every sentence.
- Open `ref.bib` and verify each `% CHECK` entry on the publisher or arXiv page. Remove any reference you cannot confirm.
- Replace the logo image with your university's logo if needed.

## Step 4: Move it to Overleaf
**Option A: upload a zip (works on every plan, recommended).**
1. Zip the *contents* of `report/` (so `main.tex` is at the top level of the zip, not inside a subfolder).
2. In Overleaf: **New Project, then Upload Project**, and choose the zip.
3. Click **Recompile**. If it fails with a `fontspec` error, open **Menu** and set **Compiler** to **XeLaTeX**.
4. To change files later, edit in Overleaf, or re-upload changed files.

**Option B: GitHub sync (paid Overleaf feature).**
Overleaf links one project to the *root* of one GitHub repository. If your report lives in a subfolder, copy it into its own new repo first, push, then in Overleaf use **New Project, then Import from GitHub**. After each push use **Menu, Sync, Pull GitHub changes**. Edits made in Overleaf go back with **Push Overleaf changes to GitHub**. Many universities give students a paid plan, so ask yours.

## Step 5: Put results in without typing them by hand
Typing numbers invites mistakes. Generate tables from your result files with a small script (for example `make_tables.py`) that writes `tables/*.tex`, and load them with `\input{tables/...}`. Re-run the script whenever the results change, then re-upload the table files.

## Common problems
| Symptom | Fix |
|---|---|
| "No PDF", fontspec error | Change Compiler to XeLaTeX, or use the version of `main.tex` that works with both engines |
| Citations show `?` | Recompile two or three times; Overleaf runs Biber automatically |
| "File not found" for an image | Check the file name and folder; names are case-sensitive |
| Zip uploads but `main.tex` is not found | The zip contains a folder instead of the files; re-zip the contents |
| Weird characters in the text | Escape `%`, `&`, `_`, `#`, `$` with a backslash in LaTeX text |

## Honesty rules (keep these when using an AI assistant)
- Never include a result that was not produced by your own runs.
- Check every citation exists and says what you claim.
- Say clearly what your study does *not* show (one dataset, small sample, one model).
- You are responsible for the final text. Understand it before you submit it.
