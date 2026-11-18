"""chiplens web app.

Run locally:   pip install -e ".[app]" && streamlit run app/streamlit_app.py
All modeling lives in src/chiplens. This file only reads inputs and draws results.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chiplens.catalog import has_spec, list_ids, load
from chiplens.scenario import PRECISION_BYTES, analyze

st.set_page_config(page_title="chiplens", page_icon=None, layout="wide")
st.title("chiplens")
st.caption(
    "Open-source tradeoff model for AI inference chips. Results are optimistic "
    "upper bounds from a roofline model, not measurements. "
    "Every input carries a source and a confidence tag."
)

chips = {cid: load("chips", cid) for cid in list_ids("chips")}
models = {mid: load("models", mid) for mid in list_ids("models")}

with st.sidebar:
    st.header("Scenario")
    chip_ids, model_ids = list(chips), list(models)
    chip_id = st.selectbox(
        "Chip",
        chip_ids,
        index=chip_ids.index("h100_sxm") if "h100_sxm" in chip_ids else 0,
        format_func=lambda i: chips[i]["name"],
    )
    model_id = st.selectbox(
        "Model",
        model_ids,
        index=model_ids.index("llama3_70b") if "llama3_70b" in model_ids else 0,
        format_func=lambda i: models[i]["name"],
    )
    chip, model = chips[chip_id], models[model_id]

    precisions = [p for p in PRECISION_BYTES if has_spec(chip, f"peak_compute_{p}")]
    precision = st.selectbox("Precision", precisions)
    sockets = st.number_input("Chips (ideal scaling)", 1, 64, 1)
    batch = st.select_slider("Batch size (users)", [2**i for i in range(0, 12)], 1)
    has_arch = has_spec(model, "layers")
    context = st.number_input(
        "Context tokens per user (KV cache)",
        0,
        131072,
        0,
        step=1024,
        disabled=not has_arch,
        help="Needs model architecture data. Disabled for models without it.",
    )

try:
    result = analyze(
        chip,
        model,
        precision=precision,
        sockets=int(sockets),
        batch=int(batch),
        context_tokens=int(context),
    )
except (ValueError, KeyError) as exc:
    st.error(str(exc))
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Speed per user", f"{result.per_user_tokens_per_s:.1f} tokens/s")
col2.metric("System throughput", f"{result.system_tokens_per_s:,.0f} tokens/s")
col3.metric("Compute utilization", f"{result.compute_utilization:.1%}")
col4.metric("Tokens per joule", f"{result.tokens_per_joule:.3f}")

placement = ", ".join(f"{gb:.1f} GB in {tier}" for tier, gb in result.placement)
st.write(
    f"Weights: **{result.weights_gb:.1f} GB** ({placement}). "
    f"Balanced batch without KV cache: **{result.balanced_batch:.0f}**."
)

if result.assumed_inputs:
    st.warning(
        "This result depends on ASSUMED inputs: "
        + ", ".join(result.assumed_inputs)
        + ". Treat it as illustrative."
    )
if result.estimated_inputs:
    st.info(
        "Estimated (derived, not stated by the source): "
        + ", ".join(result.estimated_inputs)
    )
if result.kv_gb_per_user > 0 and not result.kv_fits:
    st.error(
        f"The KV caches do not fit in memory at batch {int(batch)} "
        f"(KV per user {result.kv_gb_per_user:.2f} GB). "
        f"Largest batch that fits: {result.max_batch_by_capacity}."
    )

rows = []
for b in (2**i for i in range(0, 12)):
    try:
        r = analyze(
            chip,
            model,
            precision=precision,
            sockets=int(sockets),
            batch=b,
            context_tokens=int(context),
        )
    except ValueError:
        continue
    rows.append(
        {
            "batch": b,
            "per-user tokens/s": r.per_user_tokens_per_s,
            "system tokens/s": r.system_tokens_per_s,
        }
    )
sweep = pd.DataFrame(rows).set_index("batch")
left, right = st.columns(2)
left.subheader("Speed each user sees")
left.line_chart(sweep["per-user tokens/s"])
right.subheader("Total throughput")
right.line_chart(sweep["system tokens/s"])

st.subheader("Inputs used")
st.dataframe(
    pd.DataFrame(
        [{"input": k, "confidence": v} for k, v in result.used_inputs.items()]
    ),
    hide_index=True,
)
