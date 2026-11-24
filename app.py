import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json 

with open("titles.json", encoding="utf-8") as f:
    titles = json.load(f)

load_dotenv()


@st.cache_resource
def load_resources():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = QdrantClient(path="qdrant_data")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, client, groq_client


model, client, groq_client = load_resources()


def search(query, top_k=5):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="youtube_chunks",
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return results


def build_context(results):
    blocks = []
    for result in results:
        start = result.payload["start"]
        video_id = result.payload["video_id"]
        title = titles.get(video_id, video_id)
        minutes = int(start) // 60
        seconds = int(start) % 60
        label = f"[{title} at {minutes:02d}:{seconds:02d}]"
        blocks.append(label + "\n" + result.payload["text"])
    return "\n\n".join(blocks)


st.title("YouTube Reviser")

query = st.text_input("Ask a question")

if query:
    results = search(query, top_k=5)
    context = build_context(results)

    prompt = f"""You are a study assistant. Answer the question using ONLY the video transcript excerpts below.

Rules:
- After each point, cite its source label, like [Episode 3 –Master DSA Patterns with 2 Pointers Technique at 08:21]. Copy the label exactly as it appears in the Context.
- If the excerpts do not contain the answer, say "I couldn't find this in the videos." Do not use outside knowledge.
- The excerpts are in Hindi, but you must write your answer in the same language as the Question below. If the Question is in English, answer in English. Keep technical terms like YT-DLP, Whisper and RAG as they are.

Context:
{context}

Question:
{query}

Answer in the same language as the Question above:"""

    with st.spinner("Thinking..."):
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )

    st.write(response.choices[0].message.content)

    top = results[0].payload
    title = titles.get(top["video_id"], top["video_id"])
    url = f"https://www.youtube.com/watch?v={top['video_id']}"
    st.video(url, start_time=int(top["start"]))