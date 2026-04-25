# ============================================================
# APP.PY – CONNOISSEUR COMPANION (MAIN APPLICATION)
# ============================================================
# Run this after build.py. Provides a Gradio chat interface,
# multi‑agent orchestration, and ReAct‑style reasoning.
# ============================================================

import os, json, numpy as np
from pathlib import Path
import gradio as gr
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# CONFIGURATION – PATHS & API KEY
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable")
client = OpenAI(api_key=OPENAI_API_KEY)

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
VECDB_DIR = OUTPUT_DIR / "chroma_db"

if not VECDB_DIR.exists():
    print("❌ Vector DB not found. Run 'python build.py' first.")
    exit(1)

text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=str(VECDB_DIR), embedding_function=text_embedder)

# ============================================================
# RAG SEARCH WITH FUSION
# ============================================================
def normalize(scores):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.ones_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

def fuse(query, k=5):
    results = vector_db.similarity_search_with_score(query, k=k)
    scores = normalize([1 - s for _, s in results])
    return [(results[i][0], scores[i]) for i in range(len(results))]

# ============================================================
# MULTI‑AGENT ORCHESTRATION (PARALLEL)
# ============================================================
AGENTS = {
    "profile": "Analyze user data. Return JSON with favorite_cuisines, dietary_restrictions.",
    "trends": "Analyze restaurants. Return JSON with trends array.",
    "style": "Analyze cuisine styles. Return JSON with cuisine_analysis.",
    "nutrition": "Check dietary compliance. Return JSON with compliant_items.",
    "recs": "Synthesize into recommendations. Return JSON with recommendations array."
}

def call_agent(name, data):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": AGENTS[name]}, {"role": "user", "content": f"Data: {json.dumps(data)}"}],
        temperature=0.3
    )
    content = resp.choices[0].message.content
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content)

def run_agents():
    restaurants = [{"name": "Green Leaf", "cuisine": "Mediterranean"}, {"name": "Spice Route", "cuisine": "Indian"}]
    user_data = {"visits": ["Spice Route"], "dietary": ["vegetarian"]}
    with ThreadPoolExecutor(max_workers=4) as ex:
        profile = ex.submit(call_agent, "profile", user_data)
        trends = ex.submit(call_agent, "trends", restaurants)
        style = ex.submit(call_agent, "style", restaurants)
        nutrition = ex.submit(call_agent, "nutrition", {"restaurants": restaurants, "dietary": user_data["dietary"]})
        results = {"profile": profile.result(), "trends": trends.result(), "style": style.result(), "nutrition": nutrition.result()}
    recs = call_agent("recs", results)
    return recs.get('recommendations', [])

# ============================================================
# REACT AGENT (TOOL‑CALLING)
# ============================================================
tools = [{
    "type": "function",
    "function": {
        "name": "search_restaurants",
        "description": "Search for restaurants by query",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }
}]

def search_tool(query):
    return [{"name": r[0].metadata['name'], "score": float(r[1])} for r in fuse(query, k=3)]

def react_agent(question):
    messages = [{"role": "system", "content": "You are Connoisseur Companion. Use search_restaurants to find restaurants."},
                {"role": "user", "content": question}]
    for _ in range(5):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto")
        msg = resp.choices[0].message
        messages.append(msg.model_dump())
        if not msg.tool_calls:
            return msg.content
        for tc in msg.tool_calls:
            if tc.function.name == "search_restaurants":
                args = json.loads(tc.function.arguments)
                result = search_tool(args["query"])
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Max iterations reached."

# ============================================================
# GRADIO INTERFACE
# ============================================================
def chat_response(message, history):
    results = fuse(message, k=3)
    if results:
        reply = "🍽️ **Recommendations**\n\n"
        for i, (doc, score) in enumerate(results, 1):
            reply += f"{i}. **{doc.metadata['name']}** – {doc.metadata.get('cuisine', '')} (score: {score:.2f})\n"
        return reply
    return react_agent(message)

def agents_demo():
    recs = run_agents()
    if not recs:
        return "No recommendations generated."
    return "\n".join([f"- {r.get('name')}: {r.get('reason', '')}" for r in recs])

with gr.Blocks(title="Connoisseur Companion") as demo:
    gr.Markdown("# 🍽️ Connoisseur Companion")
    with gr.Tab("Chat"):
        gr.ChatInterface(fn=chat_response, examples=["Find me moody Italian", "cozy sushi", "romantic dinner"])
    with gr.Tab("Agents Demo"):
        gr.Markdown("Run multi‑agent orchestration (parallel)")
        btn = gr.Button("Generate Recommendations")
        out = gr.Textbox(label="Results", lines=10)
        btn.click(fn=agents_demo, outputs=out)

if __name__ == "__main__":
    demo.launch()
