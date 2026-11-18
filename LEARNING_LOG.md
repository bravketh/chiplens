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

## M1: roofline and batching

**Opening quiz** (arithmetic intensity, batch-1 idleness, doubling bandwidth):
you asked for the answers instead of attempting them (third time).

- Arithmetic intensity: ops per byte moved from memory. In decoding it is
  2B / b, so it grows with batch size.
- Batch 1: 70 GB must stream for 140 GFLOP of work, so compute is busy about
  0.94% of the time.
- Doubling bandwidth: batch-1 speed doubles (42.86 to 85.71 tokens/s), and the
  balanced batch halves (106.33 to 53.17).
- Gap: pattern of asking for answers before trying. Open item for the quiz
  after M1: answer first, even if unsure.

**Quiz after M1:** pending.

## After M1: switched to full build

You asked me to build M2 to M8 in one go and walk you through it. The M1 quiz and the
M2 to M8 gating were skipped at your request. Open items to revisit, in order:

1. Answer the M1 quiz questions yourself (flat vs falling per-user speed, 16-bit at
   6 TB/s, KV cache effect on balanced batch).
2. Predict before looking: for each milestone, guess the direction of a change before
   running it.
3. Walkthrough topics: roofline, KV cache, tiers, yield, fusion, TCO, validation.
