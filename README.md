# Advance AI Research Assistant

A modular, local-first Retrieval-Augmented Generation (RAG) application for exploring, querying, and analyzing scientific PDF documents. The application combines PDF processing, semantic and keyword retrieval, ranking, an LLM-based answer-generation pipeline, and citation-aware source navigation in a web interface.

> **Project status:** The current release implements a local RAG workflow using Ollama. Cloud-provider options (such as Gemini, Groq, and OpenAI) are architectural extension points and are not represented as implemented or tested integrations in this release.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [API Overview](#api-overview)
- [Citation and Source Traceability](#citation-and-source-traceability)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Evaluation and Limitations](#evaluation-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Advance AI Research Assistant is designed to help users work with scientific papers and other PDF documents. Users can upload and manage documents, ask research questions, and receive generated responses based on retrieved document passages. Responses include source references intended to help users trace answers back to the original PDFs.

The project is built around a modular architecture. The frontend communicates with a FastAPI backend. Document processing extracts and prepares text, chunks it, and indexes it for retrieval. The query pipeline combines semantic retrieval and keyword retrieval, fuses the results, reranks candidate passages, and supplies selected context to a language model. Citation processing maps source references to document and page-level metadata for display and PDF navigation.

### Project goals

- Provide a web-based interface for managing and querying scientific PDF documents.
- Build a practical RAG pipeline with hybrid retrieval.
- Preserve document and page-level metadata for source traceability.
- Keep the backend components modular.
- Support local LLM inference through Ollama.
- Provide an architecture that can be extended to other LLM providers.

---

## Features

### Document management

- Upload and process PDF documents.
- View managed and indexed documents.
- Rename document metadata.
- Delete documents and associated indexed data.
- Retrieve PDFs for in-app viewing.

### Research question answering

- Ask questions about the indexed document collection.
- Retrieve passages using semantic search and BM25 keyword search.
- Combine retrieval rankings using Reciprocal Rank Fusion (RRF).
- Rerank candidate passages with a Cross-Encoder.
- Construct context for LLM answer generation.
- Return answers with source references.

### Citation and source traceability

- Process citation references associated with generated answers.
- Map source information to document metadata.
- Include source identifiers, document identifiers, filenames, page numbers, and chunk identifiers in source-reference data.
- Display citations in the frontend.
- Support navigation from citations to corresponding PDF pages.

### LLM configuration

- Use Ollama for local inference.
- Retrieve and update the LLM configuration through the backend API.
- List available Ollama models.
- Keep LLM integration separated from the RAG pipeline through a provider-management abstraction.

---

## Architecture

The system is organized into frontend, API, backend services, document-processing and retrieval components, storage, and LLM integration.

### High-level components

1. **Frontend — Next.js:** Web interface for document management, research queries, answer display, citations, and PDF viewing.
2. **API — FastAPI:** HTTP interface between the frontend and backend application services.
3. **Document processing:** PDF text extraction with PyMuPDF, preprocessing, chunking, and metadata handling.
4. **Embedding generation:** Sentence Transformers embedding model used to represent document chunks and queries.
5. **Storage and indexing:** Persistent vector storage using ChromaDB and keyword retrieval using BM25; SQLite is used by the application's supporting persistence where configured.
6. **Hybrid retrieval:** Semantic retrieval and keyword retrieval run as complementary retrieval methods.
7. **Rank fusion:** Reciprocal Rank Fusion (RRF) combines retrieval result lists.
8. **Reranking:** Cross-Encoder scores query–passage pairs and reranks candidate passages.
9. **LLM integration:** Provider abstraction and local Ollama inference.
10. **Citation processing:** Source references are associated with document and page-level metadata and returned to the frontend.

### Architecture diagram

The architecture can be visualized in the project architecture notebook (`02_project_architecture.ipynb`). The notebook includes architecture and workflow diagrams covering document ingestion, retrieval, generation, and citation processing.

### Technology boundaries

The current release implements and tests the Ollama-based workflow. Gemini, Groq, and OpenAI may appear as provider options in architectural diagrams, but their integrations should be treated as future or unverified work unless separately implemented and tested.

---

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| Frontend | Next.js, JavaScript, CSS | Web application and user interface |
| Backend API | FastAPI | HTTP API and request handling |
| API data validation | Pydantic | Structured validation and response models |
| PDF processing | PyMuPDF | PDF text extraction and page-level processing |
| Embeddings | Sentence Transformers | Embedding generation for chunks and queries |
| Embedding model | `all-MiniLM-L6-v2` | Embedding model used in the project |
| Vector database | ChromaDB | Persistent semantic retrieval |
| Keyword retrieval | BM25 (`rank-bm25`) | Keyword-based passage retrieval |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) | Combines retrieval rankings |
| Reranking | Cross-Encoder | Relevance scoring and reranking |
| LLM runtime | Ollama | Local language model inference |
| Supporting persistence | SQLite (where configured) | Application metadata/persistence |
| Development | Jupyter notebooks | Project development and documentation |

Versions may vary by environment. Refer to the project environment and dependency files for the exact versions used in a particular checkout.

---

## How the RAG Pipeline Works

### 1. Document ingestion

1. The user uploads a PDF through the frontend.
2. FastAPI validates and routes the request.
3. PyMuPDF extracts text from the PDF.
4. Extracted text is prepared for processing.
5. Text is divided into chunks.
6. Chunk metadata is retained, including document and page information.
7. Embeddings are generated for the chunks.
8. Chunks and embeddings are indexed in ChromaDB; keyword indexing is prepared for BM25 retrieval.

### 2. Retrieval

When a user submits a question:

1. The question is processed as a query.
2. A query embedding is generated.
3. Semantic retrieval searches the ChromaDB vector index.
4. BM25 keyword retrieval searches the keyword index.
5. The ranked result lists are combined using RRF.

### 3. Reranking

The fused candidate passages are evaluated by the Cross-Encoder. The model scores query–passage pairs and the candidates are reordered based on their relevance scores. The selected passages are used to construct the context for answer generation.

### 4. Answer generation

The question and retrieved context are supplied to the selected Ollama model through the LLM integration layer. The model generates a response based on the provided context and the application's prompting and response-processing logic.

### 5. Citation processing

Source references are processed and associated with document metadata. The API returns answer and source information to the frontend, enabling citation display and supported PDF page navigation.

---

## Requirements

The exact requirements depend on the project environment files and the machine used to run the application.

### General requirements

- Linux, macOS, or Windows environment compatible with the project's dependencies.
- Python environment compatible with the backend dependencies.
- Node.js and npm compatible with the frontend dependencies.
- Ollama installed and running for local LLM inference.
- Sufficient RAM and storage for the embedding model, indexes, PDF files, and selected local LLM.
- A GPU is optional for some components but may improve inference or model execution when supported and configured.

### Models

The project uses the following model families/components:

- **Embedding:** `all-MiniLM-L6-v2` (384-dimensional embeddings).
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **LLM:** A compatible model installed and served by Ollama. The model name and configuration depend on the local setup.

Model availability, memory requirements, and performance depend on the hardware and model versions selected.

---

## Installation and Setup

> **Note:** This README describes the intended setup workflow. Use the actual environment files, backend requirements, frontend package files, and project configuration in your checkout as the source of truth. If setup scripts or filenames differ, follow those files.

### 1. Clone the repository

```bash
git clone <repository-url>
cd advance-ai-research-assistant
```

If you already have the project locally, open the project root instead.

### 2. Set up Python

Create or activate the project's Python environment according to its environment or dependency files.

For a Conda-based setup, an example workflow is:

```bash
conda create -n advance-ai-research-assistant python=3.11
conda activate advance-ai-research-assistant
```

Install the backend dependencies using the dependency file available in the repository. For example, if the project contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

Do not use this example command if your checkout uses a different dependency or environment file.

### 3. Set up the frontend

Navigate to the frontend directory (use the actual directory name in your checkout):

```bash
cd frontend
npm install
```

Return to the project root as needed.

### 4. Install and start Ollama

Install Ollama using the official instructions for your operating system:

https://ollama.com/

Start the Ollama service if it is not already running. Pull a model that is compatible with your local hardware. Example:

```bash
ollama pull <model-name>
```

Replace `<model-name>` with the model you intend to use. Check your project's LLM configuration for the required or default model name.

Verify Ollama and its installed models:

```bash
ollama list
```

---

## Configuration

Configuration is managed through the project's backend and/or environment configuration. Check the actual configuration files in the repository before changing values.

Typical configuration areas include:

- Backend host and port.
- Frontend API base URL.
- Ollama host/endpoint.
- Selected Ollama model.
- Document upload and storage locations.
- Index and persistence locations.

Do not commit credentials, private API keys, personal documents, or sensitive configuration files to Git.

### Environment variables

If the project uses an environment file, create the local file based on the template provided by the project, if available. Do not invent variable names: check the backend/frontend configuration code for the exact names.

---

## Running the Application

The exact commands depend on the current repository layout and scripts.

### 1. Start Ollama

Ensure Ollama is running and the desired model is available.

### 2. Start the backend

From the backend directory or project root, run the backend command defined in the project. A typical FastAPI/Uvicorn development command might look like:

```bash
uvicorn app.main:app --reload
```

Use the actual module path from the project. This command is an example, not a guarantee of the current entry point.

### 3. Start the frontend

From the frontend directory, use the script defined in `package.json`. A typical Next.js development command is:

```bash
npm run dev
```

### 4. Open the application

Open the frontend URL shown by the development server, commonly:

```text
http://localhost:3000
```

The backend may expose its API and OpenAPI documentation on a separate local port. Check the backend configuration and startup output.

---

## Using the Application

### Upload a document

1. Open the web application.
2. Use the document upload interface.
3. Select a PDF.
4. Wait for processing and indexing to finish.
5. Confirm that the document appears in the document list.

### Manage documents

Depending on the implemented UI, users can:

- View managed documents.
- Rename a document.
- Delete a document.
- Open or view a PDF.

Deleting a document should be used with care because it may remove associated indexed data.

### Ask a research question

1. Open the Research Assistant interface.
2. Enter a question related to the indexed documents.
3. Submit the question.
4. Review the generated response.
5. Inspect the citations and source information.
6. Open a citation to navigate to its source document and page when supported.

### Configure the LLM

1. Open the LLM configuration interface.
2. Review the available Ollama models.
3. Select a model supported by your local Ollama installation.
4. Save or apply the configuration.
5. Test the query workflow.

---

## API Overview

The implemented API includes the following documented endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Reports backend health. |
| GET | `/llm/config` | Returns current LLM configuration. |
| GET | `/llm/models` | Lists available Ollama models. |
| PATCH | `/llm/config` | Updates the selected LLM configuration. |
| GET | `/documents` | Lists indexed/managed documents. |
| PATCH | `/documents/{document_id}` | Renames a document's metadata. |
| DELETE | `/documents/{document_id}` | Deletes a document and associated indexed data. |
| POST | `/documents/upload` | Uploads and processes a PDF document. |
| GET | `/documents/{document_id}/pdf` | Serves a PDF for viewing, including supported legacy documents. |
| POST | `/query` | Runs the RAG query pipeline and returns an answer and source metadata. |

FastAPI's OpenAPI documentation, when enabled, can be used to inspect the request and response schemas. The API's Pydantic models define and validate structured response data.

### Source-reference fields

Source-reference data in the query response includes the following documented fields:

| Field | Description |
|---|---|
| `source_id` | Identifier for the source reference in the response. |
| `document_id` | Identifier of the associated document. |
| `filename` | Source document filename. |
| `page_number` | Source page number. |
| `chunk_id` | Identifier of the associated text chunk. |

Actual response shapes and field types should be checked against the implemented Pydantic schemas.

---

## Citation and Source Traceability

Source traceability is an important feature of the application. Retrieval and answer processing preserve source metadata so that source references can be associated with the original PDF documents and pages.

Citation processing is intended to:

- Associate answer references with retrieved sources.
- Map sources to document identifiers and filenames.
- Preserve page-level source information.
- Include chunk identifiers for source tracking.
- Return structured source information through the API.
- Support citation display and PDF navigation in the frontend.

Citations help users inspect supporting material. Generated responses should still be independently checked against the cited passages, especially for technical, scientific, or high-stakes use.

---

## Project Structure

The following is a **conceptual structure** to help navigate the project. It is not a guaranteed exhaustive or exact file tree; verify the current checkout for actual names and locations.

```text
advance-ai-research-assistant/
├── notebooks/
│   ├── 00_project_setup.ipynb
│   ├── 01_...
│   ├── 02_project_architecture.ipynb
│   └── ...
├── frontend/
│   ├── ...
│   └── ...
├── backend/
│   ├── ...
│   └── ...
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   └── figures/
├── README.md
└── ...
```

The project has been developed step by step using Jupyter notebooks. The notebooks provide development, testing, implementation notes, and architecture documentation. The production application components are organized in the frontend and backend code.

---

## Testing

Testing should be performed against the implemented code and documented accurately.

### Suggested functional checks

- [ ] Backend health endpoint responds as expected.
- [ ] LLM configuration can be retrieved.
- [ ] Available Ollama models are listed.
- [ ] LLM configuration can be updated.
- [ ] PDF upload and processing work.
- [ ] Documents are listed.
- [ ] Document metadata can be renamed.
- [ ] Document deletion removes the intended document and associated indexed data.
- [ ] PDF retrieval works for supported documents.
- [ ] Query endpoint returns an answer and structured sources.
- [ ] Semantic retrieval returns candidate passages.
- [ ] BM25 retrieval returns candidate passages.
- [ ] RRF combines retrieval rankings.
- [ ] Cross-Encoder reranking is performed on candidates.
- [ ] Citations map to the expected source metadata.
- [ ] Clicking citations navigates to the expected PDF page.
- [ ] Frontend displays errors or loading states appropriately.

### Automated tests

Run the project's automated test suite using the command and configuration present in the repository. For example, if the project uses pytest:

```bash
pytest
```

This is a typical command; confirm that tests and pytest configuration are present before using it.

---

## Evaluation and Limitations

The current application is a working RAG prototype intended for document-based research assistance. Evaluation should distinguish functional testing from quantitative retrieval or answer-quality benchmarking.

### Current limitations

- Generated answers may be incomplete, inaccurate, or unsupported even when source references are returned.
- PDF extraction quality depends on document structure, encoding, and layout.
- Scientific PDFs with complex tables, equations, multi-column layouts, or scanned pages may not be fully represented by extracted text.
- Retrieval quality depends on chunking, embeddings, query formulation, keyword indexing, and reranking.
- Local LLM performance depends on the selected model and available hardware.
- The current release uses Ollama for implemented local inference.
- Cloud integrations such as Gemini, Groq, and OpenAI should not be considered implemented or tested unless separately verified.
- The application is not a substitute for expert review or independent verification of scientific claims.

### Evaluation approach

A responsible evaluation can include:

- Functional tests of the implemented API and UI workflows.
- Inspection of retrieved passages for representative questions.
- Checking whether citations correspond to the correct document and page.
- Reviewing generated answers against their cited passages.
- Documenting failure cases and limitations.

Do not report quantitative accuracy, recall, latency, or other benchmark results unless they have been measured and documented using a defined evaluation procedure.

---

## Troubleshooting

### Backend cannot connect to Ollama

- Confirm Ollama is installed and running.
- Check that the configured Ollama host is reachable from the backend.
- Verify that the selected model is installed using `ollama list`.
- Check backend logs and LLM configuration.

### Model is unavailable

- Check the configured model name.
- Confirm the model has been pulled and is available locally.
- Consider whether the model fits the system's available RAM/VRAM.

### PDF upload or extraction fails

- Confirm the PDF file is valid and accessible.
- Review backend logs.
- Test with a simpler PDF.
- Check upload size and processing configuration.

### No relevant passages are retrieved

- Confirm the document finished processing and indexing.
- Try a more specific or differently phrased question.
- Check the semantic retrieval and BM25 outputs.
- Inspect chunk text and metadata.
- Confirm the indexes are available and consistent with the document collection.

### Citation points to an unexpected page

- Inspect the chunk's page metadata.
- Check the PDF page-number convention used in extraction and in the viewer.
- Verify the source-reference mapping and PDF navigation implementation.

### Frontend cannot connect to backend

- Confirm the backend is running.
- Verify the API base URL used by the frontend.
- Check browser console output and backend logs.
- Confirm local network or port settings.

---

## Future Work

Potential extensions include:

- Implement and test cloud LLM providers through the provider-independent interface.
- Improve PDF layout and table extraction.
- Add OCR support and improve handling of scanned documents.
- Improve chunking strategies for scientific papers.
- Evaluate retrieval and reranking quantitatively on a documented test set.
- Evaluate answer faithfulness and citation correctness.
- Improve document collection management and filtering.
- Add richer citation inspection and source previews.
- Improve application robustness, logging, and error handling.
- Explore multilingual document retrieval and answering.
- Add more comprehensive automated and integration tests.

Future work should be prioritized based on measured shortcomings and user needs.

---

## Acknowledgements

This project was developed as an applied NLP and Generative AI project, integrating PDF processing, embeddings, retrieval, reranking, local language model inference, web application development, and source traceability.

The project uses open-source libraries and model ecosystems, including FastAPI, PyMuPDF, Sentence Transformers, ChromaDB, BM25, and Ollama. Refer to the individual project dependencies and model licenses for their respective terms.

---

## License

A license has not been specified in this README. Before publicly distributing the project, add an appropriate license file and review the licenses of the dependencies and models used.

---

## Disclaimer

Advance AI Research Assistant generates responses based on retrieved document content and a language model. It may make mistakes, omit relevant details, or misinterpret source documents. Users should verify important information against the original PDFs and cited passages.
