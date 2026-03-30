import json
import os.path
import re
import urllib.request
from sentence_transformers import SentenceTransformer

LOCAL_README_PATH = "../if-engine/README.md"
GITHUB_README_URL = "https://raw.githubusercontent.com/tmanbarton/if-engine/main/README.md"
JSON_FILE_PATH = "index.json"

# Get the README either locally or from the public url
readme = None
if os.path.exists(LOCAL_README_PATH):
    with open(LOCAL_README_PATH) as f:
        readme = f.read()
else:
    with urllib.request.urlopen(GITHUB_README_URL) as r:
        readme = r.read().decode("utf-8")

# Split README on headings (#/##/###/####)
chunks = re.findall("#{1,4}(?:(?!\n#).)*", readme, re.DOTALL)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = []
for chunk, embedding in zip(chunks, embeddings):
    index.append({
        "text": chunk,
        "embedding": embedding.tolist(),
    })

with open(JSON_FILE_PATH, "w") as f:
    json.dump(index, f)
