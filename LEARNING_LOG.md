# Learning log

Your answers, the gaps we found, and how they were fixed. One section per milestone.

## M0: scaffold and data schema

**Prediction question:** If a field's `confidence` is `assumed`, what should the
app do differently when it shows that number?

- Your answer: asked for the answer instead of predicting.
- Answer given: flag it visibly, and mark any headline result that depends on it.
  Never show a guess with the same confidence as a published number.
- Gap to revisit: practice predicting before looking, since the quizzes in M1
  onward are graded on your own answers.

**Quiz after M0:** you asked for the answers instead of attempting them, so
there is nothing to grade.

- Concept: `source_url` is null only for `assumed` because a guess has no
  source, and a fake URL would make it look sourced.
- Calculation: 70 GB / 192 GB is about 0.36, so 1 chip (ceiling of the ratio).
- What if: `estimated` with no URL is rejected, since estimates must be
  retraceable. Use `assumed` if nothing can be cited.
- Gap (second time): skipped predicting. The M1 quiz comes first on purpose,
  so it is graded on your own answers.
