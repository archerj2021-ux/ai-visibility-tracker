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


# ---------- Chat panel ----------
def data_summary(df, brand):
    all_b = df["brands"].explode().dropna()
    return (
        f"Brand: {brand}\n"
        f"Overall mention rate: {df['mentioned'].mean():.0%}\n"
        f"Mention rate by model:\n{df.groupby('model')['mentioned'].mean().round(2).to_string()}\n"
        f"Mention rate by category:\n{df.groupby('category')['mentioned'].mean().round(2).to_string()}\n"
        f"Brand mention counts:\n{all_b.value_counts().to_string()}\n"
        f"Mention rate by prompt:\n{df.groupby('prompt')['mentioned'].mean().round(2).to_string()}"
    )


def rule_based_answer(q, df, brand):
    """Keyword-based answers computed directly from the data. No LLM, no API key needed."""
    q = q.lower()
    all_b = df["brands"].explode().dropna()
    if any(w in q for w in ["competitor", "rival", "who", "share of voice"]):
        top = all_b[all_b != brand].value_counts().head(3)
        lines = ", ".join(f"**{n}** ({c} mentions)" for n, c in top.items())
        return f"The most frequently recommended competitors are {lines}. Our brand has {(all_b == brand).sum()} mentions."
    if any(w in q for w in ["model", "chatgpt", "claude", "openai", "anthropic"]):
        r = df.groupby("model")["mentioned"].mean().sort_values(ascending=False)
        lines = "; ".join(f"**{m}**: {v:.0%}" for m, v in r.items())
        return f"Mention rate by model: {lines}."
    if any(w in q for w in ["categor", "weak", "strong", "worst", "best"]):
        r = df.groupby("category")["mentioned"].mean().sort_values()
        return (f"Our weakest category is **{r.index[0]}** ({r.iloc[0]:.0%} mention rate) "
                f"and our strongest is **{r.index[-1]}** ({r.iloc[-1]:.0%}).")
    if any(w in q for w in ["position", "rank", "first", "order"]):
        pos = df.loc[df["mentioned"], "position"]
        return f"When we are mentioned, we appear at an average position of **{pos.mean():.1f}** (1 = first brand named)."
    if "prompt" in q or "question" in q:
        r = df.groupby("prompt")["mentioned"].mean().sort_values()
        return f"Our weakest prompt is \"{r.index[0]}\" at {r.iloc[0]:.0%} mention rate."
    return (f"Overall we are mentioned in **{df['mentioned'].mean():.0%}** of answers. "
            "Try asking about competitors, models, categories, or our average position.")


def chat_answer(q, df, brand):
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            prompt = ("You are an analyst for an AI-visibility tracker. Answer ONLY from the data below; "
                      "if the data cannot answer, say so. Be concise.\n\n"
                      f"DATA:\n{data_summary(df, brand)}\n\nQUESTION: {q}")
            return ask_anthropic(prompt)
        except Exception as e:
            return f"LLM call failed ({e}). Falling back:\n\n" + rule_based_answer(q, df, brand)
    return rule_based_answer(q, df, brand)


st.divider()
st.subheader("Ask the data")
if os.environ.get("ANTHROPIC_API_KEY"):
    st.caption("AI assistant: answers are generated from the results above.")
else:
    st.caption("Demo assistant: answers are calculated from the results above using simple rules (no live LLM connected).")

SUGGESTED = ["Who are our top competitors?", "Which model mentions us most?", "What is our weakest category?"]
if "chat" not in st.session_state:
    st.session_state["chat"] = []

clicked = None
for col, s in zip(st.columns(len(SUGGESTED)), SUGGESTED):
    if col.button(s, use_container_width=True):
        clicked = s
question = st.chat_input("Ask about your AI visibility data...") or clicked

for m in st.session_state["chat"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question:
    st.session_state["chat"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    reply = chat_answer(question, df, brand)
    st.session_state["chat"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
