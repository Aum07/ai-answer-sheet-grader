# Sample answer sheets

Five subjects, each with one `reference.pdf` and 10–49 student sheets (5 questions each).
All student answers are synthetic (LLM-written in different quality styles) and the files
are anonymised as `student_NN.pdf`.

`manifest.csv` lists every sheet. Where the generator labelled the intended quality of a
sheet (e.g. "High Performer", "Poor Performer"), it is kept in `intended_quality`. That's
useful as a quick sanity check that predicted totals rank the sheets sensibly.

Try it: upload `ai/reference.pdf` plus a few `ai/student_*.pdf` in the web UI.
