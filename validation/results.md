# Validation: SN40L decode speed

Source of the measurements: https://arxiv.org/abs/2405.07518 (Table IV). Published values are tokens per second per user on 16 SN40L sockets with BF16 weights.

chiplens prediction: the batch-1 memory-bandwidth bound with ideal scaling across 16 sockets (HBM bandwidth and capacity multiplied by the socket count, no network cost, no KV cache). It is an upper bound, not a forecast.

| Model | Published (tok/s/user) | chiplens bound | Error | Published / bound | Audit |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama 3.1 8B | 1,042 | 1,800.0 | +72.7% | 0.58x | within bound |
| Llama 3 70B | 457 | 204.1 | -55.3% | 2.24x | exceeds bound |
| Llama 3.1 405B | 129 | 35.6 | -72.4% | 3.63x | exceeds bound |

Error is (chiplens minus published) divided by published.

## What this shows

**chiplens does not reproduce these published numbers.** The errors above are large and go in both directions.

- **Llama 3.1 8B** reaches 58% of the bound. The gap is real-system inefficiency this model ignores. Likely contributors (untested hypotheses): communication across sockets, kernel and launch overheads, and bandwidth below peak. A small model spread over many sockets amplifies all of these.
- **Llama 3 70B** beats the bandwidth bound by 2.24x. No decode that reads every weight once per token can do that, so the published run must produce more than one token per weight read. The paper states that these runs use speculative decoding (Section VI-B), which fits. This model does not include speculative decoding.
- **Llama 3.1 405B** beats the bandwidth bound by 3.63x. No decode that reads every weight once per token can do that, so the published run must produce more than one token per weight read. The paper states that these runs use speculative decoding (Section VI-B), which fits. This model does not include speculative decoding.

## Limits of this comparison

- The batch size of the published runs is not stated. chiplens assumes batch 1.
- Table values were read from an HTML rendering of the paper that repeated digits. Verify them against the PDF.
- HBM bandwidth is 1.8 TB/s in Table II and about 2 TB/s in Section IV. chiplens uses 1.8.
- The 70B row uses the Llama 3 70B model file (same dimensions as 3.1 70B) with a parameter count derived from its architecture.
- Power is not part of this comparison.
