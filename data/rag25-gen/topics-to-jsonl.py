#!/usr/bin/env python3
from pathlib import Path
import json

topics = json.loads(Path("topics/trec25_narratives_final.json").read_text())

with open("topics/trec_25.jsonl", "w") as f:
    for topic in topics:
        topic["request_id"] = topic["id"]
        topic["title"] = topic["narrative"]
        f.write(json.dumps(topic) + "\n")

