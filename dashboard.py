"""Launch: python -m streamlit run dashboard.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_engine import FEATURES, build_experiment, explain_state
from src.models.rl_agent import DOSE_ACTIONS

st.set_page_config(page_title="ChemoRL · Research studio", page_icon="◌", layout="wide")
st.markdown("""<style>
.block-container {max-width:1240px; padding-top:2.5rem;}
h1 {letter-spacing:-1.5px; font-weight:650!important;}
h2,h3 {letter-spacing:-.5px;}
[data-testid="stMetric"] {background:white; border:1px solid #e4e9ec;
 border-radius:14px; padding:18px;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stSidebar"] {border-right:1px solid #e4e9ec;}
.stButton button {border-radius:10px;}
</style>""", unsafe_allow_html=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#dbe2e6", "text.color": "#243344",
                     "axes.labelcolor": "#536477", "xtick.color": "#536477",
                     "ytick.color": "#536477", "figure.facecolor": "white"})

@st.cache_resource(show_spinner=False)
def load_run(seed, episodes):
    return build_experiment(seed=seed, episodes=episodes)

@st.cache_data(show_spinner=False)
def explain(seed, episodes, subject, cycle, action):
    return explain_state(load_run(seed, episodes), subject, cycle, action)

with st.sidebar:
    st.markdown("### ◌ ChemoRL")
    st.caption("RESEARCH STUDIO / 01")
    st.divider()
    st.markdown("**Experiment settings**")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    episodes = st.select_slider("Training episodes", options=[60, 120, 240], value=120)
    if st.button("Build simulation", type="primary", use_container_width=True):
        with st.spinner("Training DQN on synthetic states…"):
            load_run(int(seed), episodes)
        st.session_state["run_key"] = (int(seed), episodes)
    st.caption("CPU only · no dataset downloads")
    st.divider()
    st.markdown("**Data provenance**")
    st.caption("Procedural 64 × 64 images and randomized virtual profiles. No real patient records.")
    st.caption("CNN: random, frozen weights. DQN: trained in an unvalidated toy environment.")

st.caption("EXPERIMENTAL MACHINE LEARNING / POLICY INTERPRETABILITY")
st.title("A clearer view of the policy.")
st.markdown("Explore virtual trajectories, compare baselines, and inspect what drives a model score.")
st.warning("Research simulation only — not for treatment decisions. These results are not clinical evidence; the toy burden score is not a CTCAE grade and shrinkage is not a full RECIST assessment.")

if "run_key" not in st.session_state:
    st.info("Select Build simulation to train a reproducible policy and populate the dashboard with measured outputs.")
    st.stop()
run_seed, run_episodes = st.session_state["run_key"]
if (seed, episodes) != (run_seed, run_episodes):
    st.caption("Settings changed. Showing the previous run until you select Build simulation.")
run = load_run(run_seed, run_episodes)
records = run["runs"]["Dueling DQN"]
cols = st.columns(4)
cols[0].metric("Held-out virtual profiles", len(records))
cols[1].metric("Mean toy return", f"{np.mean([r['return'] for r in records]):.2f}")
cols[2].metric("Mean diameter reduction", f"{np.mean([r['shrinkage'] for r in records]):.1f}%")
cols[3].metric("Mean peak toy burden", f"{np.mean([r['peak_burden'] for r in records]):.2f}")
st.caption(f"Seed {run_seed} · {run_episodes} training episodes · {run['train_size']} training profiles · disjoint held-out cohort · one run, not a robustness study")
trajectory, attribution, comparison = st.tabs(["01 / Trajectories", "02 / SHAP explanations", "03 / Policy comparison"])

with trajectory:
    left, right = st.columns([1, 2.8], gap="large")
    with left:
        subject = st.selectbox("Virtual profile", range(len(records)), format_func=lambda i: records[i]["id"])
        rec = records[subject]
        st.image(rec["image"], caption="Procedural baseline image, not a CT scan", clamp=True, use_container_width=True)
        st.caption("The image embedding remains fixed across cycles. The renderer is not a learned progression model.")
    with right:
        st.subheader("One profile. Multiple trajectories.")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for (label, rows), color in zip(run["runs"].items(), ["#24786b", "#73869e", "#b38668", "#b9c4cb"]):
            hist = rows[subject]["history"]
            for ax, field in zip(axes, ["diameter_mm", "toxicity"]):
                ax.plot([h["cycle"] for h in hist], [h[field] for h in hist], marker="o", ms=3, color=color, label=label)
        axes[0].set(title="Simulated diameter", xlabel="Cycle", ylabel="Diameter (toy mm)")
        axes[1].set(title="Simulated burden", xlabel="Cycle", ylabel="Burden (arbitrary units)")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption("Source: current synthetic run. Early-stopped rollouts are not extrapolated.")
        st.dataframe(pd.DataFrame(rec["history"])[["cycle", "dose", "diameter_mm", "toxicity"]].rename(columns={"dose": "Toy action", "toxicity": "Toy burden"}), hide_index=True, use_container_width=True)

with attribution:
    st.subheader("Why this action score?")
    st.caption("Actual Kernel SHAP on the trained DQN. Explains a fixed action's Q-value (expected discounted toy return), not a probability or causal treatment effect.")
    c1, c2, c3 = st.columns(3)
    s = c1.selectbox("Explain profile", range(len(records)), format_func=lambda i: records[i]["id"])
    cycle = c2.selectbox("Decision cycle", list(range(len(records[s]["states"]))), format_func=lambda i: str(i + 1))
    action = c3.selectbox("Fixed action to explain", list(range(len(DOSE_ACTIONS))), index=records[s]["actions"][cycle], format_func=lambda i: f"Action {i} · {DOSE_ACTIONS[i]:.2f} (toy units)")
    with st.spinner("Computing feature coalitions…"):
        result = explain(run_seed, run_episodes, s, cycle, action)
    vals, base, out = result["values"], result["base"], result["output"]
    a, b = st.columns([2.5, 1], gap="large")
    with a:
        fig, ax = plt.subplots(figsize=(9, 4))
        order = np.argsort(-np.abs(vals))
        current = base
        for y, idx in enumerate(order):
            value = vals[idx]
            ax.barh(y, abs(value), left=min(current, current + value), height=.55,
                    color="#24786b" if value >= 0 else "#b77969")
            ax.text(max(current, current + value), y, f"  {value:+.3f}", va="center", fontsize=9)
            current += value
        ax.axvline(base, color="#9aa9b5", ls=":", label=f"Background mean {base:.3f}")
        ax.axvline(out, color="#243344", ls="--", label=f"Q-value {out:.3f}")
        ax.set_yticks(range(6), [FEATURES[i] for i in order])
        ax.invert_yaxis()
        ax.set(xlabel="Q-value (discounted toy reward units)", ylabel="Normalized state feature", title="Local SHAP waterfall")
        ax.margins(x=.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, width=900); plt.close(fig)
    with b:
        st.metric("Explained Q-value", f"{out:.4f}")
        st.metric("Background expectation", f"{base:.4f}")
        st.caption(f"Additivity residual: {result['residual']:.2e}")
        st.caption("Green raises this action score; terracotta lowers it. Neither color means clinically safe or unsafe.")
    st.caption("Background: up to 24 states sampled only from training rollouts. Six state features; 64 coalition samples. The selected image embedding is held fixed. These are not pixel-level SHAP explanations. Feature masking can create implausible combinations of correlated states.")
    st.dataframe(pd.DataFrame({"Feature": FEATURES, "Normalized value": result["state"], "SHAP (Q units)": vals}), hide_index=True, use_container_width=True)
    st.download_button("Export attribution JSON", json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}, indent=2), file_name="shap-explanation.json", mime="application/json")

with comparison:
    st.subheader("Compare behavior, not clinical efficacy.")
    summary = pd.DataFrame([{"Policy": name, "Mean return": np.mean([r["return"] for r in rows]),
                             "Mean reduction (%)": np.mean([r["shrinkage"] for r in rows]),
                             "Peak burden (mean)": np.mean([r["peak_burden"] for r in rows]),
                             "Completed six cycles (%)": 100 * np.mean([r["completed"] for r in rows])}
                            for name, rows in run["runs"].items()])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(summary["Policy"], summary["Mean return"], color=["#24786b", "#c3cdd4", "#c3cdd4", "#c3cdd4"])
    ax.set(xlabel="Mean cumulative toy reward", ylabel="Policy", title="Held-out simulation return")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(summary.round(3), hide_index=True, use_container_width=True)
    st.download_button("Export comparison CSV", summary.to_csv(index=False), "policy-comparison.csv", "text/csv")
    st.caption("All policies share the same virtual profiles and deterministic simulator. No uncertainty interval is claimed from this single training seed.")
    with st.expander("Training curve & limitations"):
        st.line_chart(pd.DataFrame({"Episode return": run["learning"]}).rename_axis("Training episode (zero-based)"))
        st.markdown("- Simulator constants and reward weights are assumptions, not calibrated patient dynamics.\n- The frozen random CNN does not demonstrate learned imaging utility.\n- No image segmentation, full RECIST, event-specific CTCAE, survival evaluation, or clinical validation.\n- SHAP explains model behavior, not whether the policy is correct.")
