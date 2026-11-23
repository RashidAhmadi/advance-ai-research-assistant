
"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./page.module.css";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: "◫" },
  { id: "documents", label: "Documents", icon: "▤" },
  { id: "assistant", label: "Research Assistant", icon: "✧" },
];

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function getSourceDetails(source) {
  const metadata = source?.metadata || {};
  return {
    filename:
      source?.filename ||
      metadata.filename ||
      source?.document_name ||
      "Research document",
    page:
      source?.page_number ??
      metadata.page_number ??
      source?.page ??
      null,
    chunkId: source?.chunk_id || metadata.chunk_id || null,
  };
}

export default function Home() {
  const [activePage, setActivePage] = useState("overview");

  const [documents, setDocuments] = useState([]);
  const [documentStats, setDocumentStats] = useState({
    document_count: 0,
    total_pages: 0,
    total_chunks: 0,
  });

  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [documentError, setDocumentError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");
  const fileInputRef = useRef(null);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [asking, setAsking] = useState(false);
  const [questionError, setQuestionError] = useState("");

  const [backendStatus, setBackendStatus] = useState("checking");

  // Document management dialog and action state.
  const [editingDocument, setEditingDocument] = useState(null);
  const [editingFilename, setEditingFilename] = useState("");
  const [savingRename, setSavingRename] = useState(false);
  const [deletingDocument, setDeletingDocument] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [managementMessage, setManagementMessage] = useState("");
  const [managementError, setManagementError] = useState("");

  async function loadDocuments() {
    setLoadingDocuments(true);
    setDocumentError("");

    try {
      const response = await fetch(`${API_URL}/documents`);

      if (!response.ok) {
        throw new Error(`Could not load documents (${response.status}).`);
      }

      const data = await response.json();

      setDocuments(data.documents || []);
      setDocumentStats({
        document_count: data.document_count || 0,
        total_pages: data.total_pages || 0,
        total_chunks: data.total_chunks || 0,
      });
    } catch (error) {
      setDocumentError(error.message || "Failed to load documents.");
    } finally {
      setLoadingDocuments(false);
    }
  }

  async function checkBackend() {
    try {
      const response = await fetch(`${API_URL}/health`);
      setBackendStatus(response.ok ? "online" : "offline");
    } catch {
      setBackendStatus("offline");
    }
  }

  useEffect(() => {
    loadDocuments();
    checkBackend();
  }, []);

  const filteredDocuments = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();

    if (!term) return documents;

    return documents.filter((document) =>
      document.filename.toLowerCase().includes(term)
    );
  }, [documents, searchTerm]);

  const uploadedDocumentCount = documents.filter(
    (document) => document.is_uploaded
  ).length;

  function handleFileSelection(event) {
    const file = event.target.files?.[0];

    setUploadMessage("");
    setUploadError("");

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setSelectedFile(null);
      setUploadError("Please select a PDF file.");
      event.target.value = "";
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setSelectedFile(null);
      setUploadError("The file exceeds the 50 MB upload limit.");
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
  }

  async function handleUpload(event) {
    event.preventDefault();

    if (!selectedFile || uploading) return;

    setUploading(true);
    setUploadMessage("");
    setUploadError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document upload failed.");
      }

      setUploadMessage(
        data.message ||
          `Successfully uploaded ${selectedFile.name}.`
      );
      setSelectedFile(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      await loadDocuments();
    } catch (error) {
      setUploadError(error.message || "Document upload failed.");
    } finally {
      setUploading(false);
    }
  }

  function openRenameDialog(document) {
    setEditingDocument(document);
    setEditingFilename(document.filename);
    setManagementMessage("");
    setManagementError("");
  }

  function closeRenameDialog() {
    if (savingRename) return;
    setEditingDocument(null);
    setEditingFilename("");
  }

  async function handleRenameDocument(event) {
    event.preventDefault();

    if (!editingDocument || savingRename) return;

    const filename = editingFilename.trim();

    if (!filename) {
      setManagementError("Please enter a filename.");
      return;
    }

    if (!filename.toLowerCase().endsWith(".pdf")) {
      setManagementError("The filename must end with .pdf.");
      return;
    }

    setSavingRename(true);
    setManagementError("");
    setManagementMessage("");

    try {
      const response = await fetch(
        `${API_URL}/documents/${encodeURIComponent(editingDocument.document_id)}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ filename }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not rename the document.");
      }

      setEditingDocument(null);
      setEditingFilename("");
      setManagementMessage(data.message || "Document renamed successfully.");
      await loadDocuments();
    } catch (error) {
      setManagementError(error.message || "Could not rename the document.");
    } finally {
      setSavingRename(false);
    }
  }

  function openDeleteDialog(document) {
    setDeletingDocument(document);
    setManagementMessage("");
    setManagementError("");
  }

  function closeDeleteDialog() {
    if (deleting) return;
    setDeletingDocument(null);
  }

  async function handleDeleteDocument() {
    if (!deletingDocument || deleting) return;

    setDeleting(true);
    setManagementError("");
    setManagementMessage("");

    try {
      const response = await fetch(
        `${API_URL}/documents/${encodeURIComponent(deletingDocument.document_id)}`,
        {
          method: "DELETE",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not delete the document.");
      }

      setDeletingDocument(null);
      setManagementMessage(
        data.message || "Document deleted successfully."
      );
      await loadDocuments();
    } catch (error) {
      setManagementError(error.message || "Could not delete the document.");
    } finally {
      setDeleting(false);
    }
  }

  async function handleQuestion(event) {
    event.preventDefault();

    const trimmedQuestion = question.trim();

    if (trimmedQuestion.length < 3 || asking) return;

    setAsking(true);
    setQuestionError("");
    setAnswer(null);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          top_k: 5,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "The research query failed.");
      }

      setAnswer(data);
    } catch (error) {
      setQuestionError(error.message || "Could not process your question.");
    } finally {
      setAsking(false);
    }
  }

  function navigate(page) {
    setActivePage(page);
    setUploadMessage("");
    setUploadError("");
  }

  return (
    <div className={styles.appShell}>
      <aside className={styles.sidebar}>
        <div className={styles.brand}>
          <div className={styles.brandMark}>A</div>
          <div>
            <div className={styles.brandName}>Advance AI</div>
            <div className={styles.brandSubtitle}>Research Assistant</div>
          </div>
        </div>

        <div className={styles.navLabel}>WORKSPACE</div>

        <nav className={styles.navigation}>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`${styles.navItem} ${
                activePage === item.id ? styles.navItemActive : ""
              }`}
              onClick={() => navigate(item.id)}
            >
              <span className={styles.navIcon}>{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className={styles.sidebarBottom}>
          <div className={styles.sidebarInfo}>
            <div className={styles.sidebarInfoIcon}>✦</div>
            <div>
              <strong>Research workspace</strong>
              <p>Organize, search and explore your scientific papers.</p>
            </div>
          </div>

          <div className={styles.sidebarFooter}>
            <span className={styles.footerDot} />
            Advance AI Research Assistant
            <span className={styles.version}>v1.0</span>
          </div>
        </div>
      </aside>

      <main className={styles.mainArea}>
        <header className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span>Workspace</span>
            <span className={styles.breadcrumbDivider}>/</span>
            <strong>
              {NAV_ITEMS.find((item) => item.id === activePage)?.label}
            </strong>
          </div>

          <div className={styles.topbarRight}>
            <div className={styles.systemStatus}>
              <span
                className={`${styles.statusDot} ${
                  backendStatus === "online"
                    ? styles.statusOnline
                    : backendStatus === "offline"
                    ? styles.statusOffline
                    : ""
                }`}
              />
              API{" "}
              {backendStatus === "checking"
                ? "Connecting"
                : backendStatus === "online"
                ? "Connected"
                : "Unavailable"}
            </div>
            <div className={styles.avatar}>RA</div>
          </div>
        </header>

        <div className={styles.content}>
          {activePage === "overview" && (
            <>
              <section className={styles.pageHeading}>
                <div>
                  <div className={styles.eyebrow}>YOUR RESEARCH WORKSPACE</div>
                  <h1>Research overview</h1>
                  <p>
                    Manage your scientific documents and discover insights
                    with AI-powered research.
                  </p>
                </div>
                <button
                  className={styles.primaryButton}
                  onClick={() => navigate("documents")}
                >
                  <span>＋</span> Add document
                </button>
              </section>

              <section className={styles.statsGrid}>
                <StatCard
                  label="Indexed documents"
                  value={formatNumber(documentStats.document_count)}
                  detail="Available in your library"
                  icon="▤"
                />
                <StatCard
                  label="Total pages"
                  value={formatNumber(documentStats.total_pages)}
                  detail="Pages represented in indexed chunks"
                  icon="▧"
                />
                <StatCard
                  label="Text chunks"
                  value={formatNumber(documentStats.total_chunks)}
                  detail="Available for semantic search"
                  icon="⌘"
                />
                <StatCard
                  label="Uploaded documents"
                  value={formatNumber(uploadedDocumentCount)}
                  detail="Added through this workspace"
                  icon="↥"
                />
              </section>

              <section className={styles.overviewGrid}>
                <div className={styles.panel}>
                  <div className={styles.panelHeader}>
                    <div>
                      <h2>Recent documents</h2>
                      <p>Your indexed research papers</p>
                    </div>
                    <button
                      className={styles.textButton}
                      onClick={() => navigate("documents")}
                    >
                      View all →
                    </button>
                  </div>

                  <DocumentList
                    documents={documents.slice(0, 5)}
                    loading={loadingDocuments}
                    error={documentError}
                  />
                </div>

                <div className={styles.panel}>
                  <div className={styles.panelHeader}>
                    <div>
                      <h2>Research assistant</h2>
                      <p>Ask questions across your papers</p>
                    </div>
                    <div className={styles.aiIcon}>✧</div>
                  </div>

                  <div className={styles.assistantPreview}>
                    <div className={styles.assistantOrb}>✧</div>
                    <h3>Explore your research</h3>
                    <p>
                      Ask a question and get an AI-generated answer grounded
                      in your indexed documents, with source references.
                    </p>
                    <button
                      className={styles.primaryButton}
                      onClick={() => navigate("assistant")}
                    >
                      Open research assistant →
                    </button>
                  </div>
                </div>
              </section>
            </>
          )}

          {activePage === "documents" && (
            <>
              <section className={styles.pageHeading}>
                <div>
                  <div className={styles.eyebrow}>KNOWLEDGE BASE</div>
                  <h1>Document library</h1>
                  <p>
                    Upload, organize and manage the scientific PDFs available
                    to your research assistant.
                  </p>
                </div>
              </section>

              <section className={styles.statsGrid}>
                <StatCard
                  label="Documents"
                  value={formatNumber(documentStats.document_count)}
                  detail="Indexed in the knowledge base"
                  icon="▤"
                />
                <StatCard
                  label="Pages"
                  value={formatNumber(documentStats.total_pages)}
                  detail="Represented in indexed chunks"
                  icon="▧"
                />
                <StatCard
                  label="Text chunks"
                  value={formatNumber(documentStats.total_chunks)}
                  detail="Searchable content units"
                  icon="⌘"
                />
              </section>

              <section className={styles.panel}>
                <div className={styles.panelHeader}>
                  <div>
                    <h2>Add research documents</h2>
                    <p>Upload a PDF to process and index its content.</p>
                  </div>
                  <span className={styles.fileLimit}>PDF · MAX 50 MB</span>
                </div>

                <form onSubmit={handleUpload}>
                  <div
                    className={styles.uploadArea}
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => {
                      event.preventDefault();
                      const file = event.dataTransfer.files?.[0];

                      if (file) {
                        const transfer = new DataTransfer();
                        transfer.items.add(file);

                        if (fileInputRef.current) {
                          fileInputRef.current.files = transfer.files;
                          handleFileSelection({
                            target: fileInputRef.current,
                          });
                        }
                      }
                    }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,application/pdf"
                      className={styles.hiddenInput}
                      onChange={handleFileSelection}
                    />
                    <div className={styles.uploadIcon}>↑</div>
                    <strong>
                      {selectedFile
                        ? selectedFile.name
                        : "Drop your PDF here or browse files"}
                    </strong>
                    <span>
                      {selectedFile
                        ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB selected`
                        : "Select a PDF document from your computer"}
                    </span>
                  </div>

                  <div className={styles.uploadActions}>
                    <span className={styles.uploadHint}>
                      Documents are processed and indexed for retrieval.
                    </span>
                    <button
                      type="submit"
                      className={styles.primaryButton}
                      disabled={!selectedFile || uploading}
                    >
                      {uploading ? "Processing document..." : "Upload and index"}
                    </button>
                  </div>
                </form>

                {uploadMessage && (
                  <div className={styles.successMessage}>
                    ✓ {uploadMessage}
                  </div>
                )}
                {uploadError && (
                  <div className={styles.errorMessage}>{uploadError}</div>
                )}
              </section>

              <section className={styles.panel}>
                <div className={styles.panelHeader}>
                  <div>
                    <h2>All documents</h2>
                    <p>
                      {formatNumber(documents.length)} documents in your library
                    </p>
                  </div>

                  <button
                    className={styles.secondaryButton}
                    onClick={loadDocuments}
                    disabled={loadingDocuments}
                  >
                    ↻ Refresh
                  </button>
                </div>

                <div className={styles.searchBox}>
                  <span>⌕</span>
                  <input
                    type="search"
                    placeholder="Search documents by filename..."
                    value={searchTerm}
                    onChange={(event) => setSearchTerm(event.target.value)}
                  />
                </div>

                {managementMessage && (
                  <div className={styles.successMessage}>
                    ✓ {managementMessage}
                  </div>
                )}
                {managementError && (
                  <div className={styles.errorMessage}>{managementError}</div>
                )}

                <DocumentList
                  documents={filteredDocuments}
                  loading={loadingDocuments}
                  error={documentError}
                  showActions
                  onEdit={openRenameDialog}
                  onDelete={openDeleteDialog}
                />
              </section>
            </>
          )}

          {activePage === "assistant" && (
            <>
              <section className={styles.pageHeading}>
                <div>
                  <div className={styles.eyebrow}>AI-POWERED DISCOVERY</div>
                  <h1>Research assistant</h1>
                  <p>
                    Ask questions about your scientific papers and receive
                    evidence-based answers with source references.
                  </p>
                </div>
                <div className={styles.modelIndicator}>
                  <span className={styles.statusDot} />
                  Local AI · Ollama
                </div>
              </section>

              <section className={styles.assistantPanel}>
                <div className={styles.assistantWelcome}>
                  <div className={styles.assistantOrbLarge}>✧</div>
                  <div className={styles.eyebrow}>YOUR RESEARCH PARTNER</div>
                  <h2>What would you like to explore?</h2>
                  <p>
                    Your question will be searched across the indexed
                    documents. The assistant will use retrieved evidence
                    to generate an answer and identify its sources.
                  </p>
                </div>

                <form
                  className={styles.questionForm}
                  onSubmit={handleQuestion}
                >
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    placeholder="For example: How do deep neural networks help identify different cell types?"
                    rows={4}
                    maxLength={2000}
                  />
                  <div className={styles.questionFormFooter}>
                    <span>{question.length}/2000 characters</span>
                    <button
                      type="submit"
                      className={styles.primaryButton}
                      disabled={asking || question.trim().length < 3}
                    >
                      {asking ? "Researching..." : "Ask research assistant  →"}
                    </button>
                  </div>
                </form>

                {questionError && (
                  <div className={styles.errorMessage}>{questionError}</div>
                )}

                {asking && (
                  <div className={styles.loadingAnswer}>
                    <div className={styles.spinner} />
                    Searching documents and generating your answer...
                  </div>
                )}

                {answer && (
                  <div className={styles.answerContainer}>
                    <div className={styles.answerHeader}>
                      <div>
                        <div className={styles.eyebrow}>GENERATED RESPONSE</div>
                        <h3>Research answer</h3>
                      </div>
                      {answer.execution_time !== undefined && (
                        <span className={styles.executionTime}>
                          {Number(answer.execution_time).toFixed(2)}s
                        </span>
                      )}
                    </div>

                    <div className={styles.answerQuestion}>
                      <strong>Your question</strong>
                      <p>{answer.question || question}</p>
                    </div>

                    <div className={styles.answerText}>
                      {answer.answer || "No answer was returned."}
                    </div>

                    {answer.citation_warning && (
                      <div className={styles.warningMessage}>
                        {answer.citation_warning}
                      </div>
                    )}

                    {Array.isArray(answer.sources) &&
                      answer.sources.length > 0 && (
                        <div className={styles.sourcesSection}>
                          <h3>Retrieved sources</h3>
                          <div className={styles.sourceList}>
                            {answer.sources.map((source, index) => {
                              const details = getSourceDetails(source);

                              return (
                                <div
                                  className={styles.sourceItem}
                                  key={
                                    source.chunk_id ||
                                    details.chunkId ||
                                    index
                                  }
                                >
                                  <div className={styles.sourceNumber}>
                                    S{index + 1}
                                  </div>
                                  <div className={styles.sourceDetails}>
                                    <strong>{details.filename}</strong>
                                    <span>
                                      {details.page !== null
                                        ? `Page ${details.page}`
                                        : "Source retrieved"}
                                      {details.chunkId
                                        ? ` · ${details.chunkId}`
                                        : ""}
                                    </span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                  </div>
                )}
              </section>
            </>
          )}
        </div>

        {editingDocument && (
          <div
            className={styles.modalBackdrop}
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) closeRenameDialog();
            }}
          >
            <section
              className={styles.modal}
              role="dialog"
              aria-modal="true"
              aria-labelledby="rename-dialog-title"
            >
              <div className={styles.modalHeader}>
                <div>
                  <div className={styles.eyebrow}>DOCUMENT MANAGEMENT</div>
                  <h2 id="rename-dialog-title">Rename document</h2>
                </div>
                <button
                  type="button"
                  className={styles.modalClose}
                  onClick={closeRenameDialog}
                  disabled={savingRename}
                  aria-label="Close dialog"
                >
                  ×
                </button>
              </div>

              <form onSubmit={handleRenameDocument}>
                <label className={styles.modalLabel} htmlFor="document-filename">
                  PDF filename
                </label>
                <input
                  id="document-filename"
                  className={styles.modalInput}
                  type="text"
                  value={editingFilename}
                  onChange={(event) => setEditingFilename(event.target.value)}
                  maxLength={255}
                  autoFocus
                />
                <p className={styles.modalHint}>
                  This changes the filename shown in your library. Indexed
                  content and document references remain unchanged.
                </p>

                {managementError && (
                  <div className={styles.errorMessage}>{managementError}</div>
                )}

                <div className={styles.modalActions}>
                  <button
                    type="button"
                    className={styles.secondaryButton}
                    onClick={closeRenameDialog}
                    disabled={savingRename}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className={styles.primaryButton}
                    disabled={savingRename || !editingFilename.trim()}
                  >
                    {savingRename ? "Saving..." : "Save changes"}
                  </button>
                </div>
              </form>
            </section>
          </div>
        )}

        {deletingDocument && (
          <div
            className={styles.modalBackdrop}
            onMouseDown={(event) => {
              if (event.target === event.currentTarget) closeDeleteDialog();
            }}
          >
            <section
              className={styles.modal}
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="delete-dialog-title"
            >
              <div className={styles.modalHeader}>
                <div>
                  <div className={styles.eyebrow}>DOCUMENT MANAGEMENT</div>
                  <h2 id="delete-dialog-title">Delete document?</h2>
                </div>
                <button
                  type="button"
                  className={styles.modalClose}
                  onClick={closeDeleteDialog}
                  disabled={deleting}
                  aria-label="Close dialog"
                >
                  ×
                </button>
              </div>

              <p className={styles.deleteWarning}>
                You are about to permanently remove{" "}
                <strong>{deletingDocument.filename}</strong> from your
                indexed library, including its chunks and vectors.
                {deletingDocument.is_uploaded
                  ? " Its uploaded PDF copy will also be deleted."
                  : " The original PDF outside the uploads folder will not be deleted."}
              </p>

              {managementError && (
                <div className={styles.errorMessage}>{managementError}</div>
              )}

              <div className={styles.modalActions}>
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={closeDeleteDialog}
                  disabled={deleting}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className={styles.dangerButton}
                  onClick={handleDeleteDocument}
                  disabled={deleting}
                >
                  {deleting ? "Deleting..." : "Delete document"}
                </button>
              </div>
            </section>
          </div>
        )}

        <footer className={styles.mainFooter}>
          <span>Advance AI Research Assistant</span>
          <span>Scientific knowledge, made accessible.</span>
        </footer>
      </main>
    </div>
  );
}

function StatCard({ label, value, detail, icon }) {
  return (
    <div className={styles.statCard}>
      <div className={styles.statTop}>
        <span>{label}</span>
        <span className={styles.statIcon}>{icon}</span>
      </div>
      <div className={styles.statValue}>{value}</div>
      <div className={styles.statDetail}>{detail}</div>
    </div>
  );
}

function DocumentList({
  documents,
  loading,
  error,
  showActions = false,
  onEdit,
  onDelete,
}) {
  if (loading) {
    return <div className={styles.emptyState}>Loading documents...</div>;
  }

  if (error) {
    return <div className={styles.errorMessage}>{error}</div>;
  }

  if (!documents.length) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyIcon}>▤</div>
        <strong>No documents found</strong>
        <p>Upload a PDF document to get started.</p>
      </div>
    );
  }

  return (
    <div className={styles.documentList}>
      {documents.map((document) => (
        <div className={styles.documentRow} key={document.document_id}>
          <div className={styles.pdfIcon}>PDF</div>

          <div className={styles.documentInfo}>
            <strong title={document.filename}>{document.filename}</strong>
            <span>
              {formatNumber(document.page_count)} pages
              <span className={styles.metaSeparator}>·</span>
              {formatNumber(document.chunk_count)} chunks
            </span>
          </div>

          <div className={styles.documentOrigin}>
            {document.is_uploaded ? "Uploaded document" : "Project document"}
          </div>

          <div className={styles.documentStatus}>
            <span className={styles.statusDot} />
            Indexed
          </div>

          {showActions && (
            <div className={styles.documentActions}>
              <button
                type="button"
                className={styles.editButton}
                onClick={() => onEdit?.(document)}
                aria-label={`Rename ${document.filename}`}
              >
                Edit
              </button>
              <button
                type="button"
                className={styles.deleteButton}
                onClick={() => onDelete?.(document)}
                aria-label={`Delete ${document.filename}`}
              >
                Delete
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}