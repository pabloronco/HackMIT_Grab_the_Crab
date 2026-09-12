# Versioned Milestone Receipts

`reports/` contains **small, reviewable, versioned evidence receipts** for accepted methodological milestones.

It is intentionally different from `data/processed/`:

- `data/processed/` is reproducible working output and remains gitignored;
- `reports/` stores only compact accepted receipts needed for team review, AI context, judging, and reproducibility provenance;
- raw source datasets and large external caches must not be copied here.

A milestone receipt should normally contain:

- a short Markdown receipt with status, command, source checksums, summary, and claim/assumption discipline;
- compact audit JSON;
- compact final table(s) needed to inspect the accepted result;
- a lightweight SVG when a visual audit materially helps.

Receipt artifacts are evidence of what the team accepted at a point in time. They are **not** model inputs and should not silently replace the reproducible pipeline that generated them.
