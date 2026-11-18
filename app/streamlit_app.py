"""chiplens web app: what limits this AI chip, and what would help?

Run locally:   pip install -e ".[app]" && streamlit run app/streamlit_app.py
All modeling lives in src/chiplens. This file reads inputs and draws results.
"""

from __future__ import annotations

import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from chiplens.audit import audit_per_user_claim, audit_total_claim
from chiplens.brief import scenario_brief
from chiplens.catalog import has_spec, list_ids, load, spec_value
from chiplens.insight import diagnose
from chiplens.lever_costs import cheapest_lever, lever_costs, unit_costs_from
from chiplens.requirements import LEVERS, meets_target, required_multipliers
from chiplens.scenario import PRECISION_BYTES, analyze, with_multipliers
from chiplens.validation import VALIDATION_DIR, load_validation, run_validation

HERE = Path(__file__).resolve().parent
ORANGE, BLUE, RED, GREY = "#e8833a", "#3b82f6", "#e5484d", "#8b8b8b"
REGIME_COLOR = {"memory": ORANGE, "compute": BLUE, "capacity": RED}
BADGE = {"published": "green", "estimated": "orange", "assumed": "red"}
BATCHES = [2**i for i in range(0, 12)]

st.set_page_config(page_title="chiplens", layout="wide")
st.markdown(
    f"<style>{(HERE / 'style.css').read_text()}</style>", unsafe_allow_html=True
)

CHIPS = {i: load("chips", i) for i in list_ids("chips")}
MODELS = {i: load("models", i) for i in list_ids("models")}

VAL_META = load_validation(VALIDATION_DIR / "data" / "sn40l_table_iv.yaml")
val_rows = run_validation(VAL_META)
LIBRARY = {r.model_id: r for r in val_rows}

STORIES = {
    "Custom": None,
    "One person chatting": {"batch": 1, "context": 2048},
    "Busy API server": {"batch": 64, "context": 4096},
    "Long-document assistant": {"batch": 8, "context": 32768},
}

st.session_state.setdefault(
    "chip", "h100_sxm" if "h100_sxm" in CHIPS else next(iter(CHIPS))
)
st.session_state.setdefault(
    "model", "llama3_70b" if "llama3_70b" in MODELS else next(iter(MODELS))
)
for key, default in (
    ("sockets", 1),
    ("batch", 1),
    ("context", 0),
    ("claim", 457.0),
    ("spec_k", 1.0),
    ("eff", 1.0),
    ("target", 50.0),
    ("lib", "llama3_70b"),
):
    st.session_state.setdefault(key, default)


def apply_story() -> None:
    story = STORIES[st.session_state["story"]]
    if story:
        st.session_state["batch"] = story["batch"]
        st.session_state["context"] = story["context"]


def load_published_claim() -> None:
    row = LIBRARY[st.session_state["lib"]]
    st.session_state.update(
        chip=VAL_META["chip"],
        model=row.model_id,
        precision=VAL_META["precision"],
        sockets=int(VAL_META["sockets"]),
        batch=1,
        context=0,
        claim=float(row.measured),
        story="Custom",
        view="Check a claim",
        spec_k=1.0,
        eff=1.0,
    )


def apply_spec_k(ratio: float) -> None:
    st.session_state["spec_k"] = float(min(8.0, max(1.0, round(ratio, 1))))


def apply_efficiency(ratio: float) -> None:
    st.session_state["eff"] = float(min(1.0, max(0.05, round(ratio, 2))))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your scenario")
    st.selectbox(
        "Start from a story",
        list(STORIES),
        key="story",
        on_change=apply_story,
        help="Pre-fills batch size and context length for a typical use.",
    )
    st.selectbox(
        "Chip", list(CHIPS), key="chip", format_func=lambda i: CHIPS[i]["name"]
    )
    st.selectbox(
        "Model", list(MODELS), key="model", format_func=lambda i: MODELS[i]["name"]
    )
    chip, model = CHIPS[st.session_state["chip"]], MODELS[st.session_state["model"]]

    precisions = [p for p in PRECISION_BYTES if has_spec(chip, f"peak_compute_{p}")]
    if st.session_state.get("precision") not in precisions:
        st.session_state["precision"] = precisions[0]
    st.selectbox(
        "Precision",
        precisions,
        key="precision",
        help="Bytes per weight. 8-bit is 1 byte, 16-bit is 2 bytes.",
    )
    st.number_input(
        "Chips working together",
        1,
        1024,
        key="sockets",
        help="Assumes perfect scaling, with no network cost.",
    )
    st.select_slider(
        "Users served at once (batch)",
        BATCHES,
        key="batch",
        help="More users share each read of the weights.",
    )
    has_arch = has_spec(model, "layers")
    st.number_input(
        "Context per user (tokens)",
        0,
        131072,
        step=1024,
        key="context",
        disabled=not has_arch,
        help="How much conversation each user keeps. Needs model architecture data.",
    )
    with st.expander("Adjust the model"):
        st.slider(
            "Tokens per weight read",
            1.0,
            8.0,
            step=0.1,
            key="spec_k",
            help="1 is plain decoding. Above 1 models speculative decoding.",
        )
        st.slider(
            "Efficiency vs ideal",
            0.05,
            1.0,
            step=0.01,
            key="eff",
            help="Real systems reach a fraction of the ideal speed.",
        )

precision = st.session_state["precision"]
sockets = int(st.session_state["sockets"])
batch = int(st.session_state["batch"])
context = int(st.session_state["context"]) if has_arch else 0


spec_k = float(st.session_state["spec_k"])
eff = float(st.session_state["eff"])


def run(c: dict, b: int = batch):
    return analyze(
        c,
        model,
        precision=precision,
        sockets=sockets,
        batch=b,
        context_tokens=context,
        tokens_per_read=spec_k,
        efficiency=eff,
    )


# ------------------------------------------------------------------- hero
st.markdown(
    '<div class="hero"><h1>chiplens</h1>'
    "<p>What limits this AI chip, and what would help?</p></div>",
    unsafe_allow_html=True,
)

try:
    result = run(chip)
except (ValueError, KeyError) as exc:
    st.error(str(exc))
    st.info("Try a smaller model, more chips, or a lower precision.")
    st.stop()

verdict = diagnose(result)
levers = "".join(f"<span>{html.escape(x)}</span>" for x in verdict.levers)
st.markdown(
    f'<div class="verdict regime-{verdict.regime}">'
    f'<div class="tag">{verdict.regime}-bound</div>'
    f"<h2>{html.escape(verdict.headline)}</h2>"
    f"<p>{html.escape(verdict.detail)}</p>"
    f'<div class="levers"><b>What would help:</b>{levers}</div></div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Speed per user", f"{result.per_user_tokens_per_s:.1f}", help="tokens/s")
m2.metric(
    "Total throughput", f"{result.system_tokens_per_s:,.0f}", help="tokens/s, all users"
)
m3.metric("Compute in use", f"{result.compute_utilization:.1%}")
m4.metric("Tokens per joule", f"{result.tokens_per_joule:.3f}")

if result.assumed_inputs:
    st.warning(
        "This result leans on ASSUMED inputs: "
        + ", ".join(result.assumed_inputs)
        + ". Treat it as illustrative."
    )
st.caption(
    "Optimistic upper bound from a roofline model, not a measurement. "
    "See the Validation tab for how far off it can be."
)

_levers = {
    "2x memory bandwidth": with_multipliers(chip, bandwidth=2.0),
    "2x compute": with_multipliers(chip, compute=2.0),
    "2x memory capacity": with_multipliers(chip, capacity=2.0),
}
_brief_rows = []
for _name, _c in _levers.items():
    _r = run(_c)
    _brief_rows.append(
        (
            _name,
            (_r.per_user_tokens_per_s / result.per_user_tokens_per_s - 1) * 100,
            (_r.system_tokens_per_s / result.system_tokens_per_s - 1) * 100,
        )
    )
st.download_button(
    "Download one-page brief (Markdown)",
    scenario_brief(
        chip,
        model,
        precision,
        result,
        verdict,
        sockets=sockets,
        context_tokens=context,
        upgrades=_brief_rows,
    ),
    file_name="chiplens_brief.md",
    mime="text/markdown",
)

NAV = [
    "Bottleneck",
    "Requirements",
    "Batching",
    "Upgrades",
    "Compare chips",
    "Check a claim",
    "Cost",
    "Validation",
    "Learn",
]
st.session_state.setdefault("view", NAV[0])
picked = st.pills(
    "View", NAV, selection_mode="single", key="view", label_visibility="collapsed"
)
view = picked or NAV[0]

# ------------------------------------------------------------- bottleneck
if view == NAV[0]:
    left, right = st.columns(2)
    color = REGIME_COLOR[verdict.regime]

    with left:
        st.subheader("Where each step's time goes")
        steps = pd.DataFrame(
            {
                "part": ["Memory reads", "Arithmetic"],
                "ms": [result.memory_step_s * 1000, result.compute_step_s * 1000],
            }
        )
        bars = (
            alt.Chart(steps)
            .mark_bar(size=34)
            .encode(
                y=alt.Y("part:N", title=None, sort=None, axis=alt.Axis(labelLimit=200)),
                x=alt.X("ms:Q", title="milliseconds per step"),
                color=alt.Color(
                    "part:N",
                    legend=None,
                    scale=alt.Scale(domain=list(steps["part"]), range=[ORANGE, BLUE]),
                ),
                tooltip=["part", alt.Tooltip("ms:Q", format=".3f")],
            )
        )
        st.altair_chart(bars, width="stretch")
        st.markdown(
            '<p class="note">The longer bar is the bottleneck. The shorter one '
            "is time that unit sits idle.</p>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Roofline: how close to the ceiling?")
        ai = 2.0 * batch / PRECISION_BYTES[precision]
        ridge = result.peak_tflops / result.effective_bandwidth_tb_s
        lo, hi = min(ai, ridge) / 8, max(ai, ridge) * 8
        xs = [lo * (hi / lo) ** (i / 60) for i in range(61)]
        roof = pd.DataFrame(
            {
                "intensity": xs,
                "tflops": [
                    min(result.peak_tflops, result.effective_bandwidth_tb_s * x)
                    for x in xs
                ],
            }
        )
        scale_x = alt.Scale(type="log", domain=[lo, hi])
        line = (
            alt.Chart(roof)
            .mark_line(color=GREY, strokeWidth=3)
            .encode(
                x=alt.X("intensity:Q", scale=scale_x, title="operations per byte read"),
                y=alt.Y("tflops:Q", scale=alt.Scale(type="log"), title="TFLOPS"),
            )
        )
        here = pd.DataFrame(
            {
                "intensity": [ai],
                "tflops": [result.achieved_tflops],
                "you": ["You are here"],
            }
        )
        dot = (
            alt.Chart(here)
            .mark_point(size=260, filled=True, color=color, opacity=1)
            .encode(
                x=alt.X("intensity:Q", scale=scale_x),
                y=alt.Y("tflops:Q", scale=alt.Scale(type="log")),
                tooltip=["you", "intensity", "tflops"],
            )
        )
        label = (
            alt.Chart(here)
            .mark_text(dy=-18, fontWeight="bold", color=color)
            .encode(x="intensity:Q", y="tflops:Q", text="you")
        )
        st.altair_chart(line + dot + label, width="stretch")
        st.markdown(
            f'<p class="note">The grey line is the most this chip can do at each '
            f"intensity. It slopes up while memory is the limit and goes flat at the "
            f"compute ceiling ({result.peak_tflops:,.0f} TFLOPS) past about "
            f"{ridge:,.0f} ops/byte.</p>",
            unsafe_allow_html=True,
        )

    st.subheader("Does it fit in memory?")
    free = result.capacity_gb - result.weights_gb - result.kv_total_gb
    parts = [
        ("Weights", result.weights_gb),
        ("KV caches (all users)", result.kv_total_gb),
        ("Free" if free >= 0 else "Overflow", abs(free)),
    ]
    cap = pd.DataFrame(parts, columns=["part", "gb"])
    cap = cap[cap["gb"] > 0].copy()
    cap["row"] = "memory"
    domain = ["Weights", "KV caches (all users)", "Free", "Overflow"]
    cap_chart = (
        alt.Chart(cap)
        .mark_bar(size=40)
        .encode(
            y=alt.Y("row:N", title=None, axis=None),
            x=alt.X("gb:Q", stack="zero", title="GB"),
            color=alt.Color(
                "part:N",
                scale=alt.Scale(domain=domain, range=[BLUE, ORANGE, "#4ade80", RED]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["part", alt.Tooltip("gb:Q", format=".1f")],
        )
        .properties(height=90)
    )
    st.altair_chart(cap_chart, width="stretch")
    placement = ", ".join(f"{gb:.1f} GB in {tier}" for tier, gb in result.placement)
    st.caption(
        f"Chip memory: {result.capacity_gb:,.0f} GB. Weights placed: {placement}."
    )

# ----------------------------------------------------------- requirements
if view == NAV[1]:
    st.subheader("Does it meet the customer's requirement?")
    st.write(
        "State what a customer needs. The users and context in the sidebar set the "
        "load. The answer is a pass or fail, and what it would take to pass."
    )
    st.number_input(
        "Required speed per user (tokens/s)",
        1.0,
        5000.0,
        step=5.0,
        key="target",
    )
    target = float(st.session_state["target"])
    got = result.per_user_tokens_per_s
    if meets_target(result, target):
        st.success(
            f"Meets it: {got:.1f} tokens/s per user against a target of {target:g} "
            f"({got / target - 1:+.0%} headroom)."
        )
    else:
        why = "" if result.kv_fits else " The KV caches also do not fit in memory."
        st.error(
            f"Misses it: {got:.1f} tokens/s per user against a target of {target:g} "
            f"({got / target - 1:+.0%}).{why}"
        )

    need = required_multipliers(
        chip,
        model,
        precision,
        target,
        sockets=sockets,
        batch=batch,
        context_tokens=context,
        tokens_per_read=spec_k,
        efficiency=eff,
    )
    st.subheader("What would close the gap?")
    st.caption("Each lever is raised on its own. The rest stay as they are.")
    labels = {
        "bandwidth": "Memory bandwidth",
        "compute": "Compute",
        "capacity": "Memory capacity",
    }
    cols = st.columns(3)
    for col, lever in zip(cols, LEVERS, strict=True):
        value = need[lever]
        if value is None:
            shown = "Will not help"
        elif value == 1.0:
            shown = "Not needed"
        else:
            shown = f"{value:.2f}x"
        col.metric(labels[lever], shown)
    unit_costs = unit_costs_from(load("assumptions", "lever_costs"))
    costs = lever_costs(need, unit_costs)
    if any(v for v in costs.values() if v):
        st.caption(
            "Extra cost per chip (ASSUMED unit costs, linear in the multiplier): "
            + ", ".join(
                f"{labels[k].lower()} "
                + ("n/a" if v is None else f"${v:,.0f}")
                for k, v in costs.items()
            )
        )
        cheap = cheapest_lever(need, unit_costs)
        if cheap:
            st.markdown(
                f'<div class="pick">Cheapest fix under these assumptions: '
                f"<b>{labels[cheap[0]].lower()}</b> at about ${cheap[1]:,.0f} per "
                "chip. The ranking depends on the assumed unit costs in "
                "data/assumptions/lever_costs.yaml, so replace them with your own."
                "</div>",
                unsafe_allow_html=True,
            )
    viable = {k: v for k, v in need.items() if v is not None and v > 1.0}
    if all(v == 1.0 for v in need.values()):
        st.markdown(
            '<div class="pick">Nothing to change: the requirement is met.</div>',
            unsafe_allow_html=True,
        )
    elif viable:
        best = min(viable, key=viable.get)
        blocked = [labels[k].lower() for k, v in need.items() if v is None]
        extra = (
            f" More {' or '.join(blocked)} alone would not fix it." if blocked else ""
        )
        st.markdown(
            f'<div class="pick">The gap closes with <b>{viable[best]:.2f}x '
            f"{labels[best].lower()}</b>.{extra}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pick">No single lever within 16x closes this gap. Change '
            "the load, the model size or the precision.</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Which chips meet it?")
    rows = []
    for c in CHIPS.values():
        try:
            r = run(c)
        except (ValueError, KeyError):
            continue
        rows.append(
            {
                "Chip": c["name"],
                "Speed per user": r.per_user_tokens_per_s,
                "Meets target": "yes" if meets_target(r, target) else "no",
                "Headroom %": (r.per_user_tokens_per_s / target - 1) * 100,
                "Trust": r.worst_confidence,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "Speed per user": st.column_config.NumberColumn(format="%.1f tok/s"),
            "Headroom %": st.column_config.NumberColumn(format="%+.0f%%"),
        },
    )

# --------------------------------------------------------------- batching
if view == NAV[2]:
    st.subheader("What changes as you add users")
    st.write(
        "Each user shares one read of the weights, so total throughput climbs almost "
        "for free until the chip runs out of arithmetic. Past that point each new "
        "user just slows everyone down."
    )
    sizes = sorted(set(BATCHES) | {batch})
    rows = []
    for b in sizes:
        r = run(chip, b)
        rows.append(
            {
                "users": b,
                "Speed per user (tokens/s)": r.per_user_tokens_per_s,
                "Total throughput (tokens/s)": r.system_tokens_per_s,
                "fits": r.kv_fits,
            }
        )
    sweep = pd.DataFrame(rows)

    def sweep_chart(col: str) -> alt.LayerChart:
        base = alt.Chart(sweep)
        ln = base.mark_line(color=GREY).encode(
            x=alt.X(
                "users:Q", scale=alt.Scale(type="log", base=2), title="users at once"
            ),
            y=alt.Y(f"{col}:Q", scale=alt.Scale(type="log"), title=col),
        )
        pts = base.mark_point(filled=True, size=55).encode(
            x="users:Q",
            y=f"{col}:Q",
            color=alt.Color(
                "fits:N",
                scale=alt.Scale(domain=[True, False], range=[GREY, RED]),
                legend=None,
            ),
            tooltip=["users", alt.Tooltip(f"{col}:Q", format=",.1f"), "fits"],
        )
        now = (
            alt.Chart(pd.DataFrame({"users": [batch]}))
            .mark_rule(color=ORANGE, strokeWidth=3)
            .encode(x="users:Q")
        )
        bal = (
            alt.Chart(pd.DataFrame({"users": [result.balanced_batch]}))
            .mark_rule(color=BLUE, strokeDash=[6, 4])
            .encode(x="users:Q")
        )
        return ln + pts + now + bal

    c1, c2 = st.columns(2)
    c1.altair_chart(sweep_chart("Speed per user (tokens/s)"), width="stretch")
    c2.altair_chart(sweep_chart("Total throughput (tokens/s)"), width="stretch")
    st.markdown(
        f'<p class="note"><b style="color:{ORANGE}">Orange line</b>: your batch '
        f'({batch}). <b style="color:{BLUE}">Blue dashed line</b>: balanced batch '
        f"(about {result.balanced_batch:.0f}, ignoring KV cache). "
        f'<b style="color:{RED}">Red dots</b>: the KV caches no longer fit in memory.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- upgrades
if view == NAV[3]:
    st.subheader("Which upgrade helps this workload most?")
    st.write(
        "A product team can only afford so much. Here each lever is doubled on its "
        "own, for the scenario in the sidebar."
    )
    options = {
        "Today": chip,
        "2x memory bandwidth": with_multipliers(chip, bandwidth=2.0),
        "2x compute": with_multipliers(chip, compute=2.0),
        "2x memory capacity": with_multipliers(chip, capacity=2.0),
    }
    outcomes = {name: run(c) for name, c in options.items()}
    base = outcomes["Today"]
    rows = []
    for name, r in outcomes.items():
        rows.append(
            {
                "Option": name,
                "Speed per user": r.per_user_tokens_per_s,
                "Total throughput": r.system_tokens_per_s,
                "Users that fit": r.max_batch_by_capacity,
                "Limit": diagnose(r).regime,
                "user_pct": (r.per_user_tokens_per_s / base.per_user_tokens_per_s - 1)
                * 100,
                "total_pct": (r.system_tokens_per_s / base.system_tokens_per_s - 1)
                * 100,
            }
        )
    table = pd.DataFrame(rows)
    shown = table.drop(columns=["user_pct", "total_pct"])
    if context == 0:
        shown = shown.drop(columns=["Users that fit"])
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            "Speed per user": st.column_config.NumberColumn(format="%.1f tok/s"),
            "Total throughput": st.column_config.NumberColumn(format="%.0f tok/s"),
        },
    )
    gains = table[table["Option"] != "Today"]
    best_user = gains.loc[gains["user_pct"].idxmax()]
    best_total = gains.loc[gains["total_pct"].idxmax()]
    if best_user["user_pct"] < 0.5 and best_total["total_pct"] < 0.5:
        verdict_text = (
            "None of these help speed here. The limit is something else, "
            "such as how many users fit in memory."
        )
    else:
        verdict_text = (
            f"For per-user speed, <b>{best_user['Option']}</b> is best "
            f"({best_user['user_pct']:+.0f}%). For total throughput, "
            f"<b>{best_total['Option']}</b> is best ({best_total['total_pct']:+.0f}%)."
        )
    st.markdown(f'<div class="pick">{verdict_text}</div>', unsafe_allow_html=True)

    melted = gains.melt(
        id_vars="Option",
        value_vars=["user_pct", "total_pct"],
        var_name="metric",
        value_name="change",
    )
    melted["metric"] = melted["metric"].map(
        {"user_pct": "Speed per user", "total_pct": "Total throughput"}
    )
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("change:Q", title="% change vs today"),
            y=alt.Y("Option:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
            yOffset="metric:N",
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(range=[ORANGE, BLUE]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Option", "metric", alt.Tooltip("change:Q", format="+.0f")],
        )
        .properties(height=190)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Scaled specs are relabelled ASSUMED: they are hypotheticals, not sourced numbers. "
        "Speeds assume the batch fits in memory."
    )

    st.subheader("Build your own what-if")
    w1, w2, w3 = st.columns(3)
    bw = w1.slider("Memory bandwidth x", 0.25, 4.0, 1.0, 0.25)
    cp = w2.slider("Compute x", 0.25, 4.0, 1.0, 0.25)
    cy = w3.slider("Memory capacity x", 0.25, 4.0, 1.0, 0.25)
    try:
        alt_r = run(with_multipliers(chip, bandwidth=bw, compute=cp, capacity=cy))
    except ValueError as exc:
        st.error(str(exc))
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(
            "Speed per user",
            f"{alt_r.per_user_tokens_per_s:.1f}",
            f"{(alt_r.per_user_tokens_per_s / base.per_user_tokens_per_s - 1) * 100:+.0f}%",
        )
        d2.metric(
            "Total throughput",
            f"{alt_r.system_tokens_per_s:,.0f}",
            f"{(alt_r.system_tokens_per_s / base.system_tokens_per_s - 1) * 100:+.0f}%",
        )
        d3.metric("Balanced batch", f"{alt_r.balanced_batch:.0f}")
        d4.metric("Limit", diagnose(alt_r).regime)

# ----------------------------------------------------------- compare chips
if view == NAV[4]:
    st.subheader("Same model, same settings, different chips")
    picks = st.multiselect(
        "Chips to compare",
        list(CHIPS),
        default=[c for c in CHIPS if c != "example_chip"],
        format_func=lambda i: CHIPS[i]["name"],
    )
    rows, skipped = [], []
    for cid in picks:
        try:
            r = run(CHIPS[cid])
        except (ValueError, KeyError) as exc:
            skipped.append(f"{CHIPS[cid]['name']}: {exc}")
            continue
        rows.append(
            {
                "Chip": CHIPS[cid]["name"],
                "Speed per user": r.per_user_tokens_per_s,
                "Total throughput": r.system_tokens_per_s,
                "Tokens per joule": r.tokens_per_joule,
                "Balanced batch": r.balanced_batch,
                "Limit": diagnose(r).regime,
                "Trust": r.worst_confidence,
            }
        )
    if rows:
        comp = pd.DataFrame(rows)
        st.dataframe(comp, hide_index=True)
        pick_metric = st.radio(
            "Chart",
            ["Speed per user", "Total throughput", "Tokens per joule"],
            horizontal=True,
        )
        st.altair_chart(
            alt.Chart(comp)
            .mark_bar()
            .encode(
                x=alt.X(f"{pick_metric}:Q"),
                y=alt.Y("Chip:N", title=None, sort="-x"),
                color=alt.Color(
                    "Limit:N",
                    scale=alt.Scale(
                        domain=list(REGIME_COLOR), range=list(REGIME_COLOR.values())
                    ),
                ),
                tooltip=[
                    "Chip",
                    alt.Tooltip(f"{pick_metric}:Q", format=",.2f"),
                    "Trust",
                ],
            )
            .properties(height=60 * len(rows) + 40),
            width="stretch",
        )
        st.caption(
            "Trust is the weakest confidence tag among the inputs used. "
            "Bars are colored by what limits each chip."
        )
    for msg in skipped:
        st.info(f"Skipped. {msg}")

# ----------------------------------------------------------- check a claim
if view == NAV[5]:
    st.subheader("Can that number be true?")
    st.write(
        "Vendors quote tokens per second. For any one user, every token needs every "
        "weight read once, so speed cannot beat "
        "**memory bandwidth divided by weight size**. Enter a claim and see."
    )
    st.selectbox(
        "Published claims",
        list(LIBRARY),
        key="lib",
        format_func=lambda i: (
            f"{LIBRARY[i].model_name}: {LIBRARY[i].measured:,.0f} tokens/s"
        ),
    )
    st.button(
        f"Load SambaNova claim: {LIBRARY[st.session_state['lib']].model_name}",
        on_click=load_published_claim,
        help="SambaNova SN40L, Table IV of the MICRO 2024 paper, 16 sockets, BF16.",
    )
    claim = st.number_input(
        "Claimed speed per user (tokens/s)", min_value=0.1, key="claim", step=10.0
    )
    audit = audit_per_user_claim(
        claim, result.effective_bandwidth_tb_s, result.weights_gb
    )
    st.caption(
        f"Checked against: {chip['name']} x{sockets}, {model['name']}, {precision}. "
        f"Bound = {audit.bound:,.1f} tokens/s per user."
    )
    gauge_max = max(claim, audit.bound) * 1.15
    gauge = pd.DataFrame(
        {
            "zone": ["Possible", "Needs an explanation"],
            "start": [0, audit.bound],
            "end": [audit.bound, gauge_max],
        }
    )
    zones = (
        alt.Chart(gauge)
        .mark_bar(size=34)
        .encode(
            x=alt.X(
                "start:Q",
                title="tokens/s per user",
                scale=alt.Scale(domain=[0, gauge_max]),
            ),
            x2="end:Q",
            color=alt.Color(
                "zone:N",
                scale=alt.Scale(
                    domain=["Possible", "Needs an explanation"], range=["#4ade80", RED]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
        )
        .properties(height=90)
    )
    marker = (
        alt.Chart(pd.DataFrame({"x": [claim], "label": ["claim"]}))
        .mark_rule(color="#ffffff", strokeWidth=4)
        .encode(x="x:Q")
    )
    st.altair_chart(zones + marker, width="stretch")
    if audit.verdict == "within bound":
        st.success(
            f"Within bound ({audit.ratio:.0%} of the limit). {audit.explanation}"
        )
        st.button(
            f"Calibrate: use {audit.ratio:.0%} as the efficiency for this chip",
            on_click=apply_efficiency,
            args=(audit.ratio,),
            help="Applies your measured fraction of the ideal speed to every view.",
        )
    else:
        st.error(f"Exceeds the bound by {audit.ratio:.2f}x. {audit.explanation}")
        st.info(
            f"One explanation: speculative decoding yielding about {audit.ratio:.1f} "
            "tokens per weight read would account for the whole gap. That is a "
            "hypothesis to put to the vendor, not a finding."
        )
        st.button(
            f"Apply {audit.ratio:.1f} tokens per read to the scenario",
            on_click=apply_spec_k,
            args=(audit.ratio,),
        )

    with st.expander("Check a total-throughput claim instead"):
        total_claim = st.number_input(
            "Claimed total tokens/s", min_value=1.0, value=10000.0
        )
        ta = audit_total_claim(
            total_claim, result.peak_tflops, spec_value(model, "params_billion")
        )
        (st.success if ta.verdict == "within bound" else st.error)(
            f"Bound {ta.bound:,.0f} tokens/s. {ta.verdict.capitalize()} "
            f"({ta.ratio:.2f}x). {ta.explanation}"
        )

# -------------------------------------------------------------------- cost
if view == NAV[6]:
    st.subheader("What does a token cost?")
    st.warning(
        "Every cost input here is ASSUMED (no public source found). Throughput is an "
        "upper bound, so cost per token is a floor and payback is optimistic."
    )
    from chiplens.report import economics_report

    c1, c2, c3 = st.columns(3)
    die_area = c1.number_input("Die area per chip (mm², your input)", 50, 1600, 800)
    n_dies = c2.number_input("Chiplets per chip", 1, 8, 1)
    overhead = c3.slider("Link area overhead per chiplet", 0.0, 0.3, 0.1)
    try:
        eco = economics_report(
            result,
            load("assumptions", "cost_example"),
            die_area_mm2=float(die_area),
            n_dies=int(n_dies),
            d2d_overhead=float(overhead),
            sockets=sockets,
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
            f"Likely range (p10 to p90): ${lo:.4f} to ${hi:.4f} per million tokens. "
            "Assumed inputs: " + ", ".join(eco["assumed_inputs"])
        )

# -------------------------------------------------------------- validation
if view == NAV[7]:
    st.subheader("How wrong is the model?")
    st.write(
        "chiplens was checked against SambaNova's published SN40L results "
        "(16 sockets, BF16)."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": r.model_name,
                    "Published (tok/s/user)": r.measured,
                    "chiplens bound": round(r.predicted, 1),
                    "Error %": round(r.error_pct, 1),
                    "Audit": r.verdict,
                }
                for r in val_rows
            ]
        ),
        hide_index=True,
    )
    st.error(
        "chiplens does not reproduce these numbers. The 70B and 405B runs beat the "
        "bandwidth bound, which the paper attributes to speculative decoding that this "
        "model omits. The 8B run reaches 58% of the bound."
    )
    chart_path = HERE.parent / "docs" / "img" / "validation.png"
    if chart_path.exists():
        st.image(str(chart_path))

# ------------------------------------------------------------------- learn
if view == NAV[8]:
    st.subheader("The ideas, in plain English")
    glossary = [
        (
            "Memory-bound vs compute-bound",
            "A chef and a waiter. The chef (compute) cooks only as fast as the waiter "
            "(memory) delivers. If the waiter is slow, the chef idles: memory-bound.",
        ),
        (
            "Batch size",
            "A waiter carrying one plate per trip wastes the trip. Serve many tables per "
            "trip (many users per weight read) and the same memory traffic feeds more work.",
        ),
        (
            "Balanced batch",
            "The number of users where the chef and the waiter are equally busy. Below "
            "it, memory limits you. Above it, compute does.",
        ),
        (
            "KV cache",
            "Each user's notebook of the conversation so far. It grows with context, "
            "takes memory, and must be read every step.",
        ),
        (
            "HBM, DDR, SRAM",
            "A desk (SRAM, tiny, instant), a shelf (HBM, small, fast), and a warehouse "
            "(DDR, huge, slow). If your files do not fit on the shelf, trips to the "
            "warehouse slow everything.",
        ),
        (
            "Yield and chiplets",
            "You pay for the whole wafer but sell only working dies. Bigger dies catch "
            "more defects, so splitting a big chip into chiplets can cost less.",
        ),
        (
            "Speculative decoding",
            "A junior drafts several words and the expert checks them in one pass. More "
            "than one token per weight read, which is how claims can beat the bound.",
        ),
        (
            "Published, estimated, assumed",
            "Every input is tagged: stated by a source, derived from stated numbers, or "
            "a guess. Results tell you which kind they lean on.",
        ),
    ]
    for term, text in glossary:
        with st.expander(term):
            st.write(text)
    st.markdown(
        "Full lesson with worked numbers: `docs/LEARNING_GUIDE.md`. "
        "Formulas and blind spots: `docs/methodology.md`."
    )
