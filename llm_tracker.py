"""
AI Visibility Tracker - prototype
Run:  pip install streamlit pandas openai anthropic
      streamlit run llm_tracker.py
Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY for live mode. Demo mode needs no keys.
"""
import os
import random
import re
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Visibility Tracker", layout="wide")

DEFAULT_PROMPTS = {
    "Category": [
        "What are the best contact center software platforms?",
        "Which cloud contact center solution is best for mid-size companies?",
        "Top omnichannel customer service platforms in 2026",
    ],
    "Comparison": [
        "Alternatives to Genesys Cloud",
        "Five9 vs Talkdesk vs NICE CXone: which is better?",
        "Best contact center software with built-in AI agent assist",
    ],
    "Buying intent": [
        "What contact center software should I choose for 200 agents?",
        "Which contact center platform has the best CRM integrations?",
    ],
}
DEFAULT_COMPETITORS = "Genesys, NICE CXone, Five9, Talkdesk, Twilio Flex, Amazon Connect, Dialpad, RingCentral, 8x8"


# ---------- LLM calls ----------
def ask_openai(prompt):
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def ask_anthropic(prompt):
    import anthropic
    r = anthropic.Anthropic().messages.create(
        model="claude-sonnet-5-5", max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


MODELS = {"ChatGPT (OpenAI)": ask_openai, "Claude (Anthropic)": ask_anthropic}


# ---------- Analysis ----------
def analyze(answer, brand, competitors):
    """Return mention flag, rank among all brands (by first appearance), and per-brand mentions."""
    brands = [brand] + competitors
    first = {}
    for b in brands:
        m = re.search(re.escape(b), answer, re.IGNORECASE)
        if m:
            first[b] = m.start()
    order = sorted(first, key=first.get)
    return {
        "mentioned": brand in first,
        "position": order.index(brand) + 1 if brand in first else None,
        "brands": order,
    }


def demo_rows(prompts, brand, competitors, n_runs):
    rows = []
    weights = {"ChatGPT (OpenAI)": 0.35, "Claude (Anthropic)": 0.25}
    for model, w in weights.items():
        for cat, p in prompts:
            for _ in range(n_runs):
                pool = random.sample(competitors, k=random.randint(3, 5))
                if random.random() < w:
                    pool.insert(random.randint(0, len(pool)), brand)
                ans = "Top options: " + ", ".join(pool)
                rows.append({"time": datetime.now(), "model": model, "category": cat,
                             "prompt": p, "answer": ans, **analyze(ans, brand, competitors)})
    return rows


def live_rows(prompts, brand, competitors, n_runs, models, progress):
    rows, total, done = [], len(prompts) * n_runs * len(models), 0
    for model in models:
        for cat, p in prompts:
            for _ in range(n_runs):
                try:
                    ans = MODELS[model](p)
                except Exception as e:
                    st.warning(f"{model} failed: {e}")
                    ans = ""
                rows.append({"time": datetime.now(), "model": model, "category": cat,
                             "prompt": p, "answer": ans, **analyze(ans, brand, competitors)})
                done += 1
                progress.progress(done / total)
    return rows


# ---------- UI ----------
st.title("AI Visibility Tracker")
st.caption("How often do LLMs recommend us, and who do they recommend instead?")

with st.sidebar:
    brand = st.text_input("Our brand", "YourBrand")
    competitors = [c.strip() for c in st.text_area("Competitors", DEFAULT_COMPETITORS).split(",") if c.strip()]
    n_runs = st.slider("Runs per prompt", 1, 5, 3)
    demo = st.toggle("Demo mode (simulated data)", value=True)
    models = st.multiselect("Models (live)", list(MODELS), default=list(MODELS), disabled=demo)
    prompts_txt = st.text_area(
        "Prompts (one per line, format: Category | prompt)",
        "\n".join(f"{c} | {p}" for c, ps in DEFAULT_PROMPTS.items() for p in ps),
        height=250,
    )
    run = st.button("Run tracking", type="primary", use_container_width=True)

prompts = [tuple(x.strip() for x in line.split("|", 1)) for line in prompts_txt.splitlines() if "|" in line]

if run:
    if demo:
        st.session_state["df"] = pd.DataFrame(demo_rows(prompts, brand, competitors, n_runs))
    else:
        bar = st.progress(0.0, text="Querying models...")
        st.session_state["df"] = pd.DataFrame(live_rows(prompts, brand, competitors, n_runs, models, bar))
        bar.empty()
    st.session_state["demo"] = demo

df = st.session_state.get("df")
if df is None:
    st.info("Configure the sidebar and click **Run tracking**. Start in demo mode to see the dashboard.")
    st.stop()

if st.session_state.get("demo"):
    st.warning("Showing simulated demo data, not real LLM output.")

# KPIs
mention_rate = df["mentioned"].mean()
avg_pos = df.loc[df["mentioned"], "position"].mean()
all_brands = df["brands"].explode().dropna()
sov = (all_brands == brand).sum() / max(len(all_brands), 1)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mention rate", f"{mention_rate:.0%}")
c2.metric("Avg. position when mentioned", f"{avg_pos:.1f}" if pd.notna(avg_pos) else "-")
c3.metric("Share of voice", f"{sov:.0%}")
c4.metric("Answers analysed", len(df))

left, right = st.columns(2)
with left:
    st.subheader("Mention rate by model")
    st.bar_chart(df.groupby("model")["mentioned"].mean())
    st.subheader("Mention rate by prompt category")
    st.bar_chart(df.groupby("category")["mentioned"].mean())
with right:
    st.subheader("Share of voice vs competitors")
    st.bar_chart(all_brands.value_counts())

st.subheader("Prompt-level results")
by_prompt = df.groupby(["category", "prompt"]).agg(
    mention_rate=("mentioned", "mean"), avg_position=("position", "mean")).reset_index()
st.dataframe(by_prompt, use_container_width=True, hide_index=True)

with st.expander("Raw answers"):
    st.dataframe(df[["time", "model", "prompt", "mentioned", "position", "answer"]], use_container_width=True)
st.download_button("Download results (CSV)", df.drop(columns=["brands"]).to_csv(index=False), "results.csv")
