# Advance AI Research Assistant — Environment Setup Guide

This guide prepares the development environment for **Advance AI Research Assistant**.
The primary development workspace is JupyterLab. The application will use FastAPI
for the backend and Next.js with JavaScript and CSS for the frontend.

> Important: You already have a working Conda environment named `nlp`. These
> instructions install into that environment. They do not create a second one.

## 1. Prerequisites

- Linux Mint / Ubuntu
- Miniconda or Anaconda
- Existing Conda environment: `nlp`
- Python 3.11
- Node.js and npm (already installed in your setup)
- Git
- Optional later: Ollama, Docker, OCR system tools

## 2. Activate and verify the existing environment

Open a terminal:

```bash
conda activate nlp
which python
python --version
```

Expected Python executable:

```text
/home/r/miniconda3/envs/nlp/bin/python
```

## 3. Install the core Python packages

From the project root (the directory containing `requirements.txt`):

```bash
python -m pip install -r requirements.txt
```

The core requirements include PDF processing, NLP, embeddings, ChromaDB,
BM25 retrieval, FastAPI, configuration, testing and code-quality tools.

### PyTorch / CUDA — do not reinstall blindly

The requirements file intentionally does not include `torch` or `torchvision`.
Your current environment already has a CUDA-enabled PyTorch installation.
A generic pip install could replace it with an incompatible build.

Verify the existing installation:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GiB):",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
```

Your current machine has an NVIDIA RTX 3050 Laptop GPU with approximately
3.68 GiB VRAM. Use small embedding batches and compact local models.

## 4. Optional packages

Do not install every optional package in advance. Install these only when
implementing the related feature:

```bash
python -m pip install -r requirements-optional.txt
```

This file contains optional cloud LLM SDKs, orchestration frameworks and OCR
Python packages. Cloud API keys are not included.

## 5. Register Jupyter kernel

Run:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name nlp --display-name "Python (nlp)"
```

Start JupyterLab from the activated environment:

```bash
conda activate nlp
jupyter lab
```

In each notebook, select **Python (nlp)** as the kernel.

## 6. spaCy language models

spaCy itself does not automatically install language models. Install only the
languages needed by your documents:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

The German model may already be installed in your environment; check before
downloading it again:

```bash
python -m spacy validate
```

## 7. Frontend: Node.js, npm and Next.js

Check the already-installed tools:

```bash
node --version
npm --version
```

When we are ready to build the frontend, create the JavaScript-based Next.js app
from the project root:

```bash
npx create-next-app@latest frontend --js --eslint --app --src-dir --use-npm
```

This is intentionally deferred until the architecture and API contract are
defined. Use JavaScript, not TypeScript.

## 8. Ollama (optional local LLM service)

The Python `ollama` package is only a client. It does not install the Ollama
server or a language model.

Install Ollama using its official instructions:
https://ollama.com/download

Then check:

```bash
ollama --version
ollama list
```

Choose a small quantized model after checking its memory requirements. The
application should also support cloud providers, but cloud API keys must remain
optional.

## 9. Optional OCR for scanned PDFs

Digital PDFs can usually be processed directly with PyMuPDF. For scanned PDFs,
OCR may be required.

On Linux Mint / Ubuntu:

```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

Then install the Python OCR dependencies:

```bash
python -m pip install pytesseract pdf2image Pillow
```

## 10. Optional Docker

Docker is not required for notebook development. Install Docker Engine and the
Compose plugin using the official guide:
https://docs.docker.com/engine/install/

Verify:

```bash
docker --version
docker compose version
```

## 11. Configure environment variables and API keys

Copy the template:

```bash
cp .env.example .env
```

Edit `.env` only when configuring a provider or local service. Never commit
`.env` or put real API keys in notebooks or source code. `.env.example` is safe
to commit because it contains placeholders only.

The default provider is Ollama. Gemini, Groq and OpenAI are optional choices.

## 12. Verify the installation

Run from the activated `nlp` environment:

```bash
python -m pip check
python -m pytest --version
node --version
npm --version
```

Then run the Python import check:

```bash
python - <<'PY'
import importlib

modules = [
    "torch", "transformers", "spacy", "chromadb",
    "sentence_transformers", "fastapi", "uvicorn",
    "pydantic", "fitz", "pypdf", "rank_bm25", "pytest"
]

for name in modules:
    try:
        importlib.import_module(name)
        print(f"OK   {name}")
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
PY
```

A warning from `pip check` about an unrelated optional package should be
investigated separately. Do not downgrade or uninstall packages just to obtain
a clean output without checking the impact.

## 13. Recreate the base Conda environment on another computer

`environment.yml` describes a minimal Python/Jupyter base environment. It does
not install machine-specific PyTorch/CUDA. For a new computer:

```bash
conda env create -f environment.yml
conda activate nlp
python -m pip install -r requirements.txt
```

Install the appropriate PyTorch build separately using:
https://pytorch.org/get-started/locally/

For your current computer, keep using the existing `nlp` environment instead
of recreating it.

## Project setup principles

- Jupyter notebooks are the primary development workspace.
- Move stable code into reusable Python modules after experiments.
- Keep LLM and embedding providers behind interchangeable interfaces.
- ChromaDB is the initial vector store; evaluate alternatives before switching.
- Keep API keys out of notebooks, source files and Git.
- Install optional dependencies only when their features are implemented.
