
"use client";

import { useState } from "react";
import styles from "./page.module.css";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");

  // Upload a scientific PDF.
  async function handleUpload(event) {
    event.preventDefault();

    if (!selectedFile) {
      setUploadStatus("Please select a PDF file.");
      return;
    }

    setUploading(true);
    setUploadStatus("");
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        `${API_URL}/documents/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "PDF upload failed.");
      }

      setUploadStatus(
        `Successfully indexed ${data.original_filename}. ` +
        `${data.pages} pages and ${data.chunks_created} chunks created.`
      );

      setSelectedFile(null);
      event.target.reset();

    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  // Submit a research question to the RAG backend.
  async function handleQuestion(event) {
    event.preventDefault();

    if (question.trim().length < 3) {
      setError("Please enter a longer research question.");
      return;
    }

    setAsking(true);
    setAnswer(null);
    setError("");

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
          top_k: 5,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Question processing failed.");
      }

      setAnswer(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <div className={styles.logo}>AI</div>
        <div>
          <h1>Advance AI Research Assistant</h1>
          <p>Scientific PDF Research and Question Answering</p>
        </div>
      </header>

      <section className={styles.intro}>
        <h2>Your Scientific Research Assistant</h2>
        <p>
          Upload scientific papers, ask research questions,
          and receive AI-generated answers supported by
          relevant source references.
        </p>
      </section>

      <section className={styles.panel}>
        <h2>1. Upload Scientific PDF</h2>
        <p className={styles.description}>
          Select a scientific PDF document to add it to
          the searchable research collection.
        </p>

        <form onSubmit={handleUpload}>
          <input
            type="file"
            accept=".pdf,application/pdf"
            onChange={(event) =>
              setSelectedFile(event.target.files[0] || null)
            }
            className={styles.fileInput}
          />

          {selectedFile && (
            <p className={styles.fileName}>
              Selected: {selectedFile.name}
            </p>
          )}

          <button
            type="submit"
            disabled={uploading || !selectedFile}
            className={styles.primaryButton}
          >
            {uploading ? "Uploading..." : "Upload and Index PDF"}
          </button>
        </form>

        {uploadStatus && (
          <p className={styles.success}>{uploadStatus}</p>
        )}
      </section>

      <section className={styles.panel}>
        <h2>2. Ask a Research Question</h2>
        <p className={styles.description}>
          Ask a question about the scientific documents
          available in the research collection.
        </p>

        <form onSubmit={handleQuestion}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Example: How do deep neural networks help identify different cell types?"
            rows={4}
            className={styles.textarea}
          />

          <button
            type="submit"
            disabled={asking || question.trim().length < 3}
            className={styles.primaryButton}
          >
            {asking ? "Researching..." : "Ask Research Assistant"}
          </button>
        </form>
      </section>

      {error && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {answer && (
        <section className={styles.result}>
          <h2>Research Answer</h2>

          <div className={styles.answerText}>
            {answer.answer}
          </div>

          <p className={styles.executionTime}>
            Execution time:{" "}
            {Number(answer.execution_time).toFixed(2)} seconds
          </p>

          {answer.citation_warning && (
            <p className={styles.warning}>
              Warning: Some citation references could not be validated.
            </p>
          )}

          <h3>Sources</h3>

          {answer.sources?.length > 0 ? (
            <ul className={styles.sourceList}>
              {answer.sources.map((source, index) => (
                <li key={`${source.chunk_id}-${index}`}>
                  <strong>[S{source.source_id}]</strong>{" "}
                  {source.filename} — Page {source.page_number}
                </li>
              ))}
            </ul>
          ) : (
            <p>No source references were returned.</p>
          )}
        </section>
      )}

      <footer className={styles.footer}>
        Advance AI Research Assistant · Scientific PDF RAG System
      </footer>
    </main>
  );
}
