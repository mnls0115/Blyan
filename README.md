# AI-Block

An experiment in running AI models on a simple blockchain structure.

## How to Run

### 1. Setup Backend

First, set up a Python virtual environment and install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize Chains

Before running the server, you must create the first "genesis" block for the meta-chain. This block tells the system which AI model to use.

```bash
python - <<'PY'
import json
from pathlib import Path
from backend.core.chain import Chain

# This will create a ./data/A directory for the meta-chain
root_dir = Path("./data")
meta_chain = Chain(root_dir, "A")

# For the first run, we use a small, fast model.
# Payloads can be any JSON, but we use this structure for the model.
spec = {"model_name": "distilbert-base-uncased"}
meta_chain.add_block(json.dumps(spec).encode())

print("✅ Meta chain initialized.")
PY
```

### 3. Run API Server

Now, you can start the FastAPI server.

```bash
uvicorn api.server:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

### 4. Run Frontend

Open a new terminal. You don't need a build step for the frontend. Simply open the `frontend/index.html` file in your web browser.

- On macOS: `open frontend/index.html`
- On Linux: `xdg-open frontend/index.html`
- On Windows: `start frontend/index.html`

You can now chat with the AI through the web interface.

## Mining New Parameters

The `/mine` endpoint allows new parameters (model weights) to be added to the parameter-chain. See `api/server.py` for the request structure. This part of the workflow is still under development. 