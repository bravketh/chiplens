# What a roofline model gets wrong about LLM inference

I built chiplens, an open-source model of AI inference chips, and then checked it
against numbers SambaNova published for its SN40L. It did not match. The ways it
failed are the useful part.

## The model in two lines

Each decoded token streams every weight from memory once and does about 2
operations per weight. So per-user speed is capped by
$\dfrac{\text{memory bandwidth}}{\text{weight size}}$, and total throughput by
$\dfrac{\text{peak compute}}{2 \times \text{parameters}}$. The batch size where the two
meet is the balanced batch. For a 70B model on a 638 TFLOPS, 3 TB/s chip at 8 bits
it is about 106: below that the chip waits on memory, and at batch 1 compute is busy
under 1% of the time.

## The test

SambaNova's MICRO 2024 paper reports tokens per second per user on 16 SN40L
sockets in BF16. With 16 x 1.8 TB/s of HBM bandwidth and ideal scaling, the
bandwidth bound at batch 1 is:

| Model | Published | chiplens bound | Error |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B | 1,042 | 1,800 | +73% |
| Llama 3.1 70B | 457 | 204 | -55% |
| Llama 3.1 405B | 129 | 36 | -72% |

## What went wrong, row by row

**70B and 405B beat the "upper bound" by 2.2x and 3.6x.** A bound you can beat is a
sign that the assumptions behind it are broken. The assumption here is one token
per read of the weights. The paper says these runs use speculative decoding: a
small model drafts several tokens and the large model checks them in one pass over
the weights. If a read yields about 2 to 4 accepted tokens, the numbers fit. The
roofline model has no way to express that, and it is exactly the kind of technique
a chip vendor's headline number can hide.

**8B reaches only 58% of the bound.** Here the model is too optimistic. A small
model spread over 16 sockets leaves little work per socket, so communication,
kernel launches and below-peak bandwidth take a large share of each step. These are
untested hypotheses. The model has no term for any of them.

## Things the simple model leaves out, ranked by how much I think they matter

1. **Tokens per weight read.** Speculative decoding and batching change what one
   read is worth.
2. **KV cache.** It adds bytes read per user per step and takes capacity. At 8K
   context a Llama 3 70B user needs about 2.7 GB of cache in 16-bit, so a 192 GB chip
   holds only about 19 users beside the weights. chiplens models this; the
   published runs do not say what batch or context they used.
3. **Multi-chip costs.** Ideal scaling ignores the network. It is the largest
   hidden assumption in the 16-socket comparison.
4. **Real utilization.** Kernels do not reach peak bandwidth or peak compute.
5. **Fusion and launch overhead.** At batch 1 the activation traffic is about
   0.03 GB next to 140 GB of weights, so fusing operations barely changes memory
   traffic. It helps through fewer kernel launches, and the paper's big fusion
   wins are on prefill and training.

## What I would trust it for

Ratios and directions, not absolutes. Doubling memory bandwidth doubles batch-1
speed and halves the balanced batch. Weights that spill from fast to slow memory
slow decode by an order of magnitude. A chip's headline speed claim above its
bandwidth bound needs an explanation, and the claim auditor asks for it.

## What I would not trust it for

Absolute tokens per second, cost per token, or payback. Cost inputs are all
assumed, and throughput is an upper bound, so cost per token is a floor.

## Limits of this check

Batch size is not stated for the published table. I read the table values from an
HTML rendering of the paper that repeated digits, so they should be verified against
the PDF. HBM bandwidth is 1.8 TB/s in one table and about 2 TB/s in the text.

Code, data with sources, and the validation report: see the repository README.
