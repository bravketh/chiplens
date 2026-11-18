"""chiplens web app.

Run locally:   pip install -e ".[app]" && streamlit run app/streamlit_app.py
All modeling lives in src/chiplens. This file only reads inputs and draws results.
"""

from __future__ import annotations

import altair as alt
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
col1.metric("Speed per user", f"{result.per_user_tokens_per_s:.1f}", help="tokens/s")
col2.metric(
    "System throughput",
    f"{result.system_tokens_per_s:,.0f}",
    help="tokens/s, all users",
)
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


def log_chart(column: str) -> alt.Chart:
    """Line chart with log axes, so small batches and big ones are both readable."""
    return (
        alt.Chart(sweep.reset_index())
        .mark_line(point=True)
        .encode(
            x=alt.X("batch:Q", scale=alt.Scale(type="log", base=2)),
            y=alt.Y(f"{column}:Q", scale=alt.Scale(type="log")),
        )
    )


left.subheader("Speed each user sees (tokens/s, log-log)")
left.altair_chart(log_chart("per-user tokens/s"), use_container_width=True)
right.subheader("Total throughput (tokens/s, log-log)")
right.altair_chart(log_chart("system tokens/s"), use_container_width=True)

st.subheader("Inputs used")
st.dataframe(
    pd.DataFrame(
        [{"input": k, "confidence": v} for k, v in result.used_inputs.items()]
    ),
    hide_index=True,
)

st.divider()
with st.expander("Economics (every cost input is assumed)"):
    from chiplens.report import economics_report

    die_area = st.number_input("Die area per chip (mm^2, your input)", 50, 1600, 800)
    n_dies = st.number_input("Chiplets per chip", 1, 8, 1)
    overhead = st.slider("Die-to-die area overhead per chiplet", 0.0, 0.3, 0.1)
    try:
        eco = economics_report(
            result,
            load("assumptions", "cost_example"),
            die_area_mm2=float(die_area),
            n_dies=int(n_dies),
            d2d_overhead=float(overhead),
            sockets=int(sockets),
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        lo, mid, hi = eco["cost_per_million_tokens_p10_p50_p90"]
        e1, e2, e3 = st.columns(3)
        e1.metric("Silicon cost per chip", f"${eco['silicon_cost']:,.0f}")
        e2.metric("Cost per million tokens (p50)", f"${mid:.4f}")
        e3.metric("Payback (p50)", f"{eco['payback_months_p10_p50_p90'][1]:.1f} months")
        st.caption(
            f"p10 to p90 cost per million tokens: ${lo:.4f} to ${hi:.4f}. "
            "Throughput is an upper bound, so cost per token is a floor and "
            "payback is optimistic. Assumed inputs: " + ", ".join(eco["assumed_inputs"])
        )

with st.expander("Validation against published SN40L results"):
    from chiplens.validation import (
        VALIDATION_DIR,
        load_validation,
        run_validation,
    )

    meta = load_validation(VALIDATION_DIR / "data" / "sn40l_table_iv.yaml")
    val_rows = run_validation(meta)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "model": r.model_name,
                    "published tok/s/user": r.measured,
                    "chiplens bound": round(r.predicted, 1),
                    "error %": round(r.error_pct, 1),
                    "audit": r.verdict,
                }
                for r in val_rows
            ]
        ),
        hide_index=True,
    )
    st.warning(
        "chiplens does not reproduce these numbers. Two published runs beat the "
        "bandwidth bound, which the paper attributes to speculative decoding that "
        "this model omits. See validation/results.md."
    )
