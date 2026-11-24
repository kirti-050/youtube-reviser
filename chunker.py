import json
from pathlib import Path

CHUNK_SECONDS = 60

def make_chunks(words, video_id):
    chunks = []
    start_index = 0
    for i, w in enumerate(words):
        if w["end"] - words[start_index]["start"] >= CHUNK_SECONDS:
            chunk_words = words[start_index: i + 1]
            text = "".join(x["word"] for x in chunk_words).strip()
            chunks.append({"video_id": video_id, "text": text,
                           "start": chunk_words[0]["start"], "end": chunk_words[-1]["end"]})
            start_index = i + 1
    if start_index < len(words):
        chunk_words = words[start_index:]
        text = "".join(x["word"] for x in chunk_words).strip()
        chunks.append({"video_id": video_id, "text": text,
                       "start": chunk_words[0]["start"], "end": chunk_words[-1]["end"]})
    return chunks

Path("chunks").mkdir(exist_ok=True)

for path in Path("transcripts").glob("*.json"):
    with open(path, encoding="utf-8") as f:
        words = json.load(f)
    chunks = make_chunks(words, path.stem)
    with open(f"chunks/{path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(path.stem, len(chunks), "chunks")