"""Send the sample session to a running prediction service.

Usage:
    python predict_test.py                      # http://localhost:9696
    python predict_test.py http://localhost:8080
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696").rstrip("/")
session = json.loads((Path(__file__).parent / "sample_session.json").read_text(encoding="utf-8"))

with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
    print("GET /health ->", response.read().decode())

request = urllib.request.Request(
    f"{base_url}/predict",
    data=json.dumps(session).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        print("POST /predict ->", json.dumps(json.loads(response.read()), indent=2))
except urllib.error.HTTPError as exc:
    print("POST /predict failed:", exc.code, exc.read().decode())
    sys.exit(1)
