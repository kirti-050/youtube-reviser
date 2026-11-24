import json
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
client = QdrantClient(path="qdrant_data")

if client.collection_exists("youtube_chunks"):
    client.delete_collection("youtube_chunks")
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
    print(path.stem, len(points), "indexed")

print(client.count("youtube_chunks"))
client.close()