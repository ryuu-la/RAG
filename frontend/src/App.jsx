import { useEffect, useMemo, useRef, useState } from "react";
import {
  deleteDocument,
  getDocumentMetrics,
  getIngestStatus,
  getModelUploads,
  getQueryMetrics,
  getSources,
  sendQuery,
  uploadToModel,
  uploadToRag,
} from "./api";

function newSession() {
  return { id: crypto.randomUUID(), title: "New chat", messages: [] };
}

const modelOptions = [
  "openai/gpt-oss-120b",
  "llama-3.3-70b-versatile",
  "llama-3.1-8b-instant",
  "mixtral-8x7b-32768",
];

function renderMarkdown(text) {
  if (!text) return null;
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return <span key={i}>{part}</span>;
  });
}

function Citations({ citations }) {
  const [open, setOpen] = useState(false);
  if (!citations || citations.length === 0) return null;

  const shortName = (src) => {
    const name = src.replace(/^[a-f0-9-]+_/, "");
    return name.length > 30 ? name.slice(0, 28) + "..." : name;
  };

  return (
    <div className="citations-wrap">
      <button className="citations-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="citations-arrow" style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)" }}>{"\u25B6"}</span>
        {citations.length} source{citations.length > 1 ? "s" : ""} cited
      </button>
      {open && (
        <div className="citations">
          {citations.map((c, i) => (
            <span key={`${c.chunk_id}-${i}`} className="chip">{shortName(c.source)} p.{c.page ?? "?"}</span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [statsOpen, setStatsOpen] = useState(false);
  const [sessions, setSessions] = useState([newSession()]);
  const [activeId, setActiveId] = useState(() => sessions[0]?.id);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [selectedModel, setSelectedModel] = useState(modelOptions[0]);
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [ragMetrics, setRagMetrics] = useState([]);
  const [modelUploads, setModelUploads] = useState([]);
  const [selectedModelUploads, setSelectedModelUploads] = useState([]);
  const [lastQueryMetrics, setLastQueryMetrics] = useState(null);
  const [notice, setNotice] = useState("");
  const [uploadJob, setUploadJob] = useState(null);
  const [showRagMetrics, setShowRagMetrics] = useState(true);
  const ragInputRef = useRef(null);
  const modelInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const activeSession = sessions.find((s) => s.id === activeId) || sessions[0];

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const totals = useMemo(() => {
    const totalDocs = ragMetrics.length;
    const totalTokens = ragMetrics.reduce((a, b) => a + (b.estimated_tokens || 0), 0);
    const totalChunks = ragMetrics.reduce((a, b) => a + (b.chunk_count || 0), 0);
    const indexedChunks = ragMetrics.reduce((a, b) => a + (b.indexed_chunk_count || 0), 0);
    return { totalDocs, totalTokens, totalChunks, indexedChunks };
  }, [ragMetrics]);

  useEffect(() => { refreshRagMetrics(); refreshModelUploads(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [activeSession?.messages?.length]);

  async function refreshRagMetrics() {
    try {
      const sources = await getSources();
      const metrics = await Promise.all(sources.map(async (s) => { try { return await getDocumentMetrics(s.doc_id); } catch { return null; } }));
      setRagMetrics(metrics.filter(Boolean));
    } catch { /* */ }
  }

  async function refreshModelUploads() { try { setModelUploads(await getModelUploads()); } catch { /* */ } }

  function pushMessage(sessionId, msg) {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== sessionId) return s;
        const next = [...s.messages, msg];
        const title = s.title === "New chat" && next.length ? String(next[0].content).slice(0, 40) : s.title;
        return { ...s, messages: next, title };
      })
    );
  }

  async function handleSend() {
    const q = input.trim();
    if (!q || isSending) return;
    setInput("");
    setIsSending(true);
    setNotice("");
    const sid = activeSession.id;
    pushMessage(sid, { role: "user", content: q });
    try {
      const out = await sendQuery({ question: q, selected_model: selectedModel, model_upload_ids: selectedModelUploads });
      pushMessage(sid, { role: "assistant", content: out.answer, citations: out.citations || [], latencyMs: out.latency_ms });
      if (out.query_id) { try { setLastQueryMetrics(await getQueryMetrics(out.query_id)); } catch { /* */ } }
    } catch (err) {
      pushMessage(sid, { role: "assistant", content: `Error: ${err.message}` });
    } finally { setIsSending(false); }
  }

  async function handleUploadToRag(file) {
    if (!file) return;
    setUploadJob({ filename: file.name, progress: 0, state: "uploading", message: "Uploading..." });
    try {
      const { job_id, doc_id } = await uploadToRag(file);
      setUploadJob((p) => ({ ...p, progress: 5, state: "processing", message: "Processing..." }));
      let status = await getIngestStatus(job_id);
      while (status.state === "queued" || status.state === "processing") {
        setUploadJob({ filename: file.name, progress: status.progress || 0, state: status.state, message: status.message });
        await new Promise((r) => setTimeout(r, 500));
        status = await getIngestStatus(job_id);
      }
      if (status.state !== "indexed") throw new Error(status.message || "Ingestion failed");
      setUploadJob({ filename: file.name, progress: 100, state: "indexed", message: status.message });
      const metric = await getDocumentMetrics(doc_id);
      setRagMetrics((prev) => [metric, ...prev.filter((x) => x.doc_id !== doc_id)]);
      setTimeout(() => setUploadJob(null), 2000);
    } catch (err) {
      setUploadJob({ filename: file.name, progress: 0, state: "failed", message: err.message });
      setTimeout(() => setUploadJob(null), 4000);
    }
  }

  async function handleUploadToModel(file) {
    if (!file) return;
    setNotice(`Uploading "${file.name}" to model...`);
    try {
      const out = await uploadToModel(file);
      await refreshModelUploads();
      setSelectedModelUploads((prev) => [...new Set([...prev, out.upload_id])]);
      setNotice(`Attached: ${file.name}`);
    } catch (err) { setNotice(`Failed: ${err.message}`); }
  }

  function toggleModelUpload(id) {
    setSelectedModelUploads((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  }

  async function handleDeleteDoc(docId) {
    try {
      await deleteDocument(docId);
      setRagMetrics((prev) => prev.filter((d) => d.doc_id !== docId));
    } catch (err) {
      setNotice(`Delete failed: ${err.message}`);
    }
  }

  return (
    <div className="app-root">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? "" : "closed"}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={() => { const s = newSession(); setSessions((p) => [s, ...p]); setActiveId(s.id); }}>
            + New chat
          </button>
          <button className="icon-btn sidebar-close-btn" onClick={() => setSidebarOpen(false)} title="Close sidebar">
            {"\u2190"}
          </button>
        </div>
        <div className="history-list">
          {sessions.map((s) => (
            <button key={s.id} className={`history-item ${s.id === activeId ? "active" : ""}`} onClick={() => setActiveId(s.id)}>
              {s.title}
            </button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <div className="model-label">Model</div>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {modelOptions.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="main-area">
        {/* Top bar */}
        <div className="main-topbar">
          <div className="main-topbar-left">
            {!sidebarOpen && (
              <button className="icon-btn" onClick={() => setSidebarOpen(true)} title="Open sidebar">{"\u2630"}</button>
            )}
          </div>
          <div className="main-topbar-right">
            <button className="icon-btn" onClick={() => setTheme((t) => t === "dark" ? "light" : "dark")} title="Toggle theme">
              {theme === "dark" ? "\u2600" : "\u263E"}
            </button>
            <button className="pill-btn" onClick={() => setStatsOpen((s) => !s)}>
              {statsOpen ? "Close" : `Stats (${totals.totalDocs})`}
            </button>
          </div>
        </div>

        {uploadJob && (
          <div className="upload-bar">
            <div className="upload-bar-top">
              <span className="upload-bar-name">{uploadJob.filename}</span>
              <span className="upload-bar-pct">{uploadJob.progress}%</span>
            </div>
            <div className="progress-track"><div className="progress-fill" style={{ width: `${uploadJob.progress}%` }} /></div>
            <div className="upload-bar-msg">{uploadJob.message}</div>
          </div>
        )}

        <div className="messages">
          <div className="messages-inner">
            {activeSession.messages.length === 0 && (
              <div className="empty-state">
                <h2>RAG Studio</h2>
                <p>Upload a PDF to RAG, then ask questions about it.</p>
              </div>
            )}
            {activeSession.messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className="msg-inner">
                  <div className="msg-avatar">{m.role === "user" ? "U" : "R"}</div>
                  <div className="msg-content">
                    <div className="msg-text">{renderMarkdown(m.content)}</div>
                    <Citations citations={m.citations} />
                    {m.latencyMs != null && <div className="msg-meta">{m.latencyMs}ms</div>}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Composer ── */}
        <div className="composer-area">
          <div className="composer-inner">
            {selectedModelUploads.length > 0 && (
              <div className="selected-files">
                {selectedModelUploads.map((id) => {
                  const item = modelUploads.find((x) => x.upload_id === id);
                  return <button key={id} className="file-chip" onClick={() => toggleModelUpload(id)}>{item?.filename || id} &times;</button>;
                })}
              </div>
            )}
            <div className="composer-box">
              <div className="plus-wrap">
                <button className="plus-btn" onClick={() => setShowPlusMenu((s) => !s)}>+</button>
                {showPlusMenu && (
                  <div className="plus-menu">
                    <button onClick={() => { setShowPlusMenu(false); ragInputRef.current?.click(); }}>Upload to RAG</button>
                    <button onClick={() => { setShowPlusMenu(false); modelInputRef.current?.click(); }}>Upload to Model</button>
                  </div>
                )}
              </div>
              <input className="text-input" value={input} onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                placeholder="Message RAG Studio..." />
              <button className="send-btn" onClick={handleSend} disabled={isSending}>{isSending ? "..." : "\u2191"}</button>
            </div>
            {notice && <div className="notice">{notice}</div>}
            <div className="composer-footer">RAG Studio -- answers strictly from your uploaded documents only.</div>
          </div>
          <input ref={ragInputRef} type="file" accept=".pdf" hidden onChange={(e) => { handleUploadToRag(e.target.files?.[0]); e.target.value = ""; }} />
          <input ref={modelInputRef} type="file" accept=".pdf,.txt,.md" hidden onChange={(e) => { handleUploadToModel(e.target.files?.[0]); e.target.value = ""; }} />
        </div>
      </div>

      {/* ── Stats Drawer ── */}
      <div className={`stats-drawer ${statsOpen ? "open" : ""}`}>
        <div className="drawer-header">
          <span className="drawer-title">Stats</span>
          <button className="icon-btn" onClick={() => setStatsOpen(false)} title="Close stats">{"\u2715"}</button>
        </div>
        <button className="section-toggle" onClick={() => setShowRagMetrics((s) => !s)}>
          <span className="section-arrow" style={{ transform: showRagMetrics ? "rotate(90deg)" : "rotate(0deg)" }}>{"\u25B6"}</span>
          RAG Metrics
        </button>

        {showRagMetrics && (
          <>
            <div className="stat-grid">
              <Stat title="Documents" value={totals.totalDocs} />
              <Stat title="Total Tokens" value={totals.totalTokens.toLocaleString()} />
              <Stat title="Chunks" value={totals.totalChunks.toLocaleString()} />
              <Stat title="Indexed" value={totals.indexedChunks.toLocaleString()} />
            </div>

            <h4>Model Uploads</h4>
            <div className="list">
              {modelUploads.length === 0 && <p style={{ color: "var(--text-dim)", fontSize: 12 }}>None yet</p>}
              {modelUploads.map((u) => (
                <label key={u.upload_id} className="list-item">
                  <input type="checkbox" checked={selectedModelUploads.includes(u.upload_id)} onChange={() => toggleModelUpload(u.upload_id)} />
                  <span>{u.filename}</span>
                  <small>{u.estimated_tokens.toLocaleString()} tok</small>
                </label>
              ))}
            </div>

            <h4>Indexed Documents</h4>
            <div className="list">
              {ragMetrics.length === 0 && <p style={{ color: "var(--text-dim)", fontSize: 12 }}>No documents indexed</p>}
              {ragMetrics.map((d) => (
                <div className="list-item doc-row" key={d.doc_id}>
                  <div className="doc-row-top">
                    <strong>{d.filename.replace(/^[a-f0-9-]+_/, "")}</strong>
                    <button className="delete-btn" onClick={() => handleDeleteDoc(d.doc_id)} title="Delete document">&times;</button>
                  </div>
                  <small>pages: {d.page_count} | ocr: {d.pages_ocr_used ?? 0} | tokens: {d.estimated_tokens.toLocaleString()} | chunks: {d.chunk_count}</small>
                </div>
              ))}
            </div>

            {lastQueryMetrics && (
              <>
                <h4>Last Query</h4>
                <div className="list-item">
                  <div>Retrieved: {lastQueryMetrics.retrieved_chunks} chunks</div>
                  <small>context: {lastQueryMetrics.context_tokens_sent.toLocaleString()} tok | response: {lastQueryMetrics.response_tokens ?? "-"} tok</small>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Stat({ title, value }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{title}</span>
      <strong className="stat-value">{value}</strong>
    </div>
  );
}
