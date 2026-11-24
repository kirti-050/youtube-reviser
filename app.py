import os
import json
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

with open("titles.json", encoding="utf-8") as f:
    titles = json.load(f)


def build_index(client, model):
    client.create_collection(
        collection_name="youtube_chunks",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    for path in Path("chunks").glob("*.json"):
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)

        embeddings = model.encode([c["text"] for c in chunks])

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['video_id']}-{i}")),
                vector=embeddings[i].tolist(),
                payload=c,
            )
            for i, c in enumerate(chunks)
        ]
        client.upsert(collection_name="youtube_chunks", points=points)


@st.cache_resource(show_spinner="Setting up the search index (first start only)...")
def load_resources():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    client = QdrantClient(path="qdrant_data")

    if not client.collection_exists("youtube_chunks"):
        build_index(client, model)

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
    video_id = top["video_id"]
    start = int(top["start"])
    title = titles.get(video_id, video_id)

    st.subheader(title)
    st.caption(f"Starts at {start // 60:02d}:{start % 60:02d}")
    st.video(f"https://www.youtube.com/watch?v={video_id}", start_time=start)