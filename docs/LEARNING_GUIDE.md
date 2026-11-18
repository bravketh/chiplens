# chiplens learning guide

Read this top to bottom once. It teaches what you need to explain the project to
a hiring manager or chip engineer: the ideas, the numbers behind them, and what
the project does and does not prove. Every number here comes from the code and tests.

## 0. The one idea

When an AI chip generates text, it is almost always **waiting for data, not
calculating**. Most of chip product thinking is about which wait to shorten:
memory speed, memory size, or arithmetic speed. chiplens is a calculator for that
question, with every input traced to a source.

## 1. Vocabulary (each term once)

| Term | Plain meaning |
| --- | --- |
| **Token** | A small piece of text, about 3 to 4 characters. A chip "generates tokens". |
| **Weights / parameters** | The numbers that make up the model. A "70B" model has 70 billion. |
| **Precision** | Bytes per weight. 8-bit (fp8, int8) is 1 byte, 16-bit (bf16) is 2 bytes. |
| **FLOP / TFLOPS** | One arithmetic operation. TFLOPS is a trillion per second: how fast the chip calculates. |
| **Memory bandwidth** | Bytes per second the chip can read from memory, in TB/s. How fast the waiter brings food. |
| **HBM** | Stacked memory right next to the chip. Fast, small, expensive. |
| **DDR** | Ordinary memory modules. Slow, big, cheap. |
| **SRAM** | Tiny memory on the chip itself. Fastest, smallest. |
| **Decode** | Generating tokens one after another. This is what the project models. |
| **Prefill** | Reading the user's prompt first. Different behavior, not modeled. |
| **Batch** | How many users the chip serves at once. |
| **KV cache** | Per-user memory of everything said so far, so the model need not recompute it. |
| **Die / wafer / yield** | A chip's silicon rectangle, the round slice it is cut from, and the share of dies that work. |
| **Chiplet** | One of several small dies packaged into a single chip. |
| **COGS / TCO** | Cost of goods sold (to make the chip). Total cost of ownership (to buy and run it). |

## 2. The roofline: kitchen analogy

A chef (compute) can only cook as fast as a waiter (memory) brings ingredients.
If the waiter is slow, the chef stands idle. If the waiter is fast, the chef is the
limit. A chip is the same.

For one generated token the chip must:
- read **every weight once** from memory, and
- do **2 operations per weight** (a multiply and an add).

Worked example, a 70B model at 8-bit (1 byte per weight) on a chip with 3 TB/s and 638 TFLOPS:

- Weights: $W = 70 \times 1 = 70$ GB.
- If memory were the only limit: $T_{mem} = \dfrac{1000 \times 3}{70} = 42.86$ tokens/s.
- If compute were the only limit: $T_{cmp} = \dfrac{1000 \times 638}{2 \times 70} = 4557$ tokens/s.

The chip is limited by the slower one: **42.86 tokens/s**.

**Arithmetic intensity** is operations done per byte read, in ops/byte. For decode it
is $\dfrac{2B}{b}$, where $B$ is batch size and $b$ is bytes per weight. A low
number means the chip is memory-bound. The **ridge point**,
$\dfrac{F}{BW} = \dfrac{638}{3} = 212.7$ ops/byte, is where it flips to compute-bound.

## 3. Batching: why the chip sits idle

At batch 1 the chip reads 70 GB to do 140 GFLOP of work. Memory needs 23 ms
($70 / 3000$ s); the arithmetic needs 0.22 ms. Compute is busy
$\dfrac{42.86}{4557} \approx 0.94\%$ of the time.

Batching fixes this. A batch of $B$ users **shares one read of the weights**, so you
do $B$ times the work for the same memory traffic. The **balanced batch** is where
the two limits meet:

$$B^{*} = \frac{T_{cmp}}{T_{mem}} = \frac{4557}{42.86} = 106.3$$

Below it you are memory-bound, above it compute-bound. Per-user speed is flat until
$B^{*}$ and then falls as $\dfrac{T_{cmp}}{B}$. Total throughput rises until $B^{*}$
and then caps.

**Doubling bandwidth:** batch-1 speed doubles (42.86 to 85.71 tokens/s) and
$B^{*}$ halves (106.3 to 53.2). Memory got faster, so compute becomes the limit sooner.
**Doubling compute:** batch-1 speed does not change at all, and $B^{*}$ doubles.

PM lesson: which upgrade is worth money depends on the workload. A single-user chat
product is memory-bound, so extra TFLOPS buys nothing there.

## 4. The KV cache: the hidden cost of conversations

Every user keeps keys and values for every layer and every past token:

$$\text{bytes per token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot b$$

For Llama 3 70B ($L=80$, $n_{kv}=8$, $d_{head}=128$, 2 bytes): 327,680 bytes per
token, so **2.68 GB per user at 8K context**. Two consequences:

1. **Capacity:** a chip with 192 GB and 140 GB of weights has 52 GB free, so it fits
   $\lfloor 52 / 2.68 \rfloor = 19$ users.
2. **Speed:** each step reads $W + B \cdot KV$ bytes, so big batches become
   memory-bound again even above $B^{*}$.

This is the biggest thing the first, simple version ignored. A model without it
overstates how many users a chip can serve.

## 5. Memory tiers: capacity cliffs

Weights fill the fastest tier first and spill into slower ones. Time per token is the
sum over tiers:

$$t = \sum_i \frac{GB_i}{1000 \cdot BW_i}$$

Example, one SN40L socket (64 GB HBM at 1.8 TB/s, 1.5 TB DDR at 0.2 TB/s) running a
141 GB model: 64 GB in HBM and 77 GB in DDR gives $0.0356 + 0.3855 = 0.42$ s, or
**2.4 tokens/s**. If it all fit in HBM it would be 12.7 tokens/s. Spilling to slow
memory costs about 5x. Capacity is a speed spec, not just a size spec.

## 6. Power

$$E_{token} = \frac{P}{T} \text{ (joules per token)}, \qquad \text{tokens per joule} = \frac{T}{P}$$

A 700 W chip making 4557 tokens/s uses 0.154 J per token. Datacenters are power-limited,
so a rack with a 40 kW budget holds $\lfloor 40000 / (700 \times 1.5) \rfloor = 38$ chips
(1.5 covers the host, network and cooling share). Efficiency decides how much you can
deploy per megawatt.

## 7. Die cost, yield, chiplets

You pay for the whole wafer but sell only the good dies:

$$\text{cost per good die} = \frac{C_{wafer}}{N_{dies} \cdot Y}, \qquad Y_{Murphy} = \left(\frac{1 - e^{-AD}}{AD}\right)^{2}$$

$A$ is die area in cm$^2$ and $D$ is defects per cm$^2$. Big dies lose twice: fewer fit
on the wafer, and each is more likely to catch a defect.

Worked example (\$10,000 wafer, 300 mm, $D = 0.1$):

| Design | Cost |
| --- | ---: |
| One 600 mm$^2$ die | \$196 |
| Two 300 mm$^2$ chiplets | \$136 |
| Two chiplets with 10% extra area for the links | \$156 |
| One 100 mm$^2$ die | \$17.25 |
| Two 55 mm$^2$ chiplets | \$17.68 (worse) |

Chiplets win for big chips and lose for small ones. That is a real product decision.

## 8. Operator fusion

An unfused model writes every intermediate result to memory and reads it back, and
each kernel launch costs a little time. Fusing keeps intermediates on chip.

At batch 1 the extra traffic is 0.03 GB against 140 GB of weights, so fusion barely
changes decode memory traffic. At batch 512 it is 16 GB. In the worked example the
step falls from 56.8 ms to 47.1 ms (1.21x). The SN40L paper's large fusion gains are
on prefill and training, which this project does not model. Takeaway: **a technique's
value depends on the regime**.

## 9. Economics

For a \$30,000 chip kept 4 years, drawing 700 W:

- Amortized purchase: $30000 / (4 \times 8760) = \$0.856$ per hour.
- Upkeep (10% a year) plus electricity: $\$0.415$ per hour.
- Total: $\$1.27$ per hour. At 4557 tokens/s and 50% busy that is **\$0.155 per million
  tokens**. Idle time still costs, so halving utilization doubles the cost per token.
- Payback at \$0.60 per million tokens: about 9.1 months.

**Monte Carlo** reruns the calculation thousands of times with uncertain inputs drawn
from ranges, then reports p10, p50 and p90. Use it to say "likely between X and Y",
not to produce one false-precision number.

Caution: all cost inputs are `assumed`, and throughput is an upper bound, so cost per
token is a floor and payback is optimistic.

## 10. Validation: the most important lesson

I compared the model with SambaNova's published SN40L results (16 sockets, BF16):

| Model | Published | chiplens bound | Error |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B | 1,042 | 1,800 | +73% |
| Llama 3 70B | 457 | 204 | -55% |
| Llama 3.1 405B | 129 | 36 | -72% |

**The model does not reproduce them**, and the two failures point in opposite directions.

- **70B and 405B beat an "upper bound".** A bound that gets beaten means one of its
  assumptions is false. The assumption here is one token per read of the weights. The
  paper says these runs use **speculative decoding**: a small model drafts several
  tokens and the big model checks them in one pass over the weights. If a read yields 2
  to 4 accepted tokens, the numbers fit.
- **8B reaches only 58% of the bound.** Real systems lose time to communication across
  sockets, launch overhead and below-peak bandwidth. These are my hypotheses, not
  tested.

The **claim auditor** turns this into a tool: a per-user claim above
$\dfrac{1000 \cdot BW}{W}$ needs an explanation (speculative decoding, smaller
effective weights, a different batch, or an error).

What you can honestly say: the model is good for **directions and ratios**, not
absolutes, and its failures are informative.

## 11. How it ties to product management and engineering

| PM skill | Where it shows up |
| --- | --- |
| Spec prioritization | What-if: which of bandwidth, compute or capacity helps this workload most |
| Workload-driven requirements | Chat is memory-bound, bulk serving is compute-bound |
| Competitive analysis | The claim auditor checks a rival's number before it goes in a deck |
| Pricing and build decisions | Chiplet vs monolithic, cost per token |
| Communicating uncertainty | Every number tagged published, estimated or assumed |

| Engineering skill | Where it shows up |
| --- | --- |
| Performance modeling | Pure functions with units in docstrings |
| Testing | Hand-worked expected values, written before the code |
| Data provenance | YAML with source, date and confidence per field |
| Validation | A committed report that says where the model is wrong, with a test that keeps it current |
| Delivery | CI, tags, changelog |

## 11b. The PM workflow in the app

1. **Requirements:** type what the customer needs (say 50 tokens/s per user). The app says
   pass or fail and gives the smallest bandwidth, compute or capacity multiplier that
   passes. At batch 1 an H100 misses 50 by 5%; 1.05x bandwidth closes it and extra compute
   does nothing. An H200 clears it with 36% headroom.
2. **Upgrades:** which doubling helps this workload.
3. **Claims:** paste a vendor number. If it beats the bandwidth bound, the implied tokens
   per weight read is the hypothesis to put to the vendor (speculative decoding).
4. **Calibrate:** if you have a measured number, its ratio to the model is your
   efficiency, applied to every view.
5. **Brief:** download a one-page Markdown summary to paste into a doc.

## 12. What to say in an interview

- "I built an inference chip model, validated it against SambaNova's published numbers,
  and found it was off by +73%, -55% and -72%. The 70B and 405B results beat the
  bandwidth bound, which pointed to speculative decoding that a roofline model cannot
  express."
- "It tells you which upgrade matters for which workload, and it flags when a claim is
  physically implausible."
- "Every input has a source and a confidence tag. All cost inputs are assumed, and the
  tool says so."

Do **not** claim: that it predicts real tokens per second, that cost figures are real,
or that it covers more than two chips. Know the limits in section 13 cold.

## 13. Honest limits

- Absolute speeds are upper bounds. Cost per token is a floor. Payback is optimistic.
- Multi-chip results assume perfect scaling with no network cost.
- Prefill, speculative decoding, mixture-of-experts and attention arithmetic are not modeled.
- SN40L power (600 W) is an `assumed` placeholder. H100 dense compute is `estimated`
  (half the sparse figure). All cost inputs are `assumed`.
- The SN40L table values came from an HTML rendering of the paper that repeated digits,
  so verify them against the PDF. The batch size of those runs is not stated.

## 14. Test yourself (answers below)

1. Why does doubling memory bandwidth halve the balanced batch?
2. A chip has 4 TB/s and 400 TFLOPS at 8 bits, running a 70B model. What are the
   batch-1 speed and the balanced batch?
3. What happens to the KV cache per user, and to the users that fit, if context doubles?
4. Why can two small chiplets cost less than one big die, but not always?
5. A vendor claims 457 tokens/s per user where the bound says 205. Name two explanations.
6. Why does fusing operations barely help at batch 1?

### Answers

1. $B^{*} = \dfrac{F \cdot b}{2 \cdot BW}$, so it is inversely proportional to bandwidth.
   Faster memory means compute becomes the limit at a smaller batch.
2. $T_{mem} = \dfrac{1000 \times 4}{70} = 57.14$ tokens/s, $T_{cmp} = \dfrac{1000 \times 400}{140} = 2857$,
   so $B^{*} = \dfrac{2857}{57.14} = 50$.
3. KV per user doubles, and the number of users that fit in the free memory roughly
   halves.
4. Smaller dies yield much better and fit more per wafer, so silicon cost drops. But
   the links between chiplets cost extra area and packaging. Small dies already yield
   well, so the overhead outweighs the saving.
5. Speculative decoding (more than one token per weight read), smaller effective weights
   (quantization, sparsity, mixture of experts), a different batch or precision than
   assumed, or an error in the claim.
6. The activation traffic (about 0.03 GB) is tiny next to the weights (140 GB), so
   memory traffic barely changes. Fusion helps at larger batch and through fewer kernel
   launches.

## 15. Where things live

| Idea | Code | Test |
| --- | --- | --- |
| Roofline, batching | `src/chiplens/roofline.py` | `tests/test_roofline.py` |
| Tiers, KV cache | `memory_tiers.py`, `llm.py` | `test_memory_tiers.py`, `test_llm.py` |
| Power | `power.py` | `test_power.py` |
| Die cost | `diecost.py` | `test_diecost.py` |
| Fusion | `fusion.py` | `test_fusion.py` |
| Economics | `economics.py`, `report.py` | `test_economics.py` |
| Claim auditor, validation | `audit.py`, `validation.py` | `test_audit.py`, `test_validation.py` |
| Verdict, what-if | `insight.py`, `scenario.py` | `test_insight.py`, `test_whatif.py` |
| Formulas and blind spots | `docs/methodology.md` | |
| Why each choice | `docs/decisions.md` | |

Read the test docstrings: each one works the expected number out by hand.
