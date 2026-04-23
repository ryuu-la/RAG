import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  deleteDocument,
  getDocumentMetrics,
  getExportUrl,
  getIngestStatus,
  getModelUploads,
  getSources,
  streamQuery,
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
  "gemma-4-31b-it",
];

const TOOL_LABELS = {
  search_documents: { icon: "🔍", label: "Searching documents" },
  lookup_document: { icon: "📄", label: "Looking up document" },
  web_search: { icon: "🌐", label: "Searching the web" },
  export_csv: { icon: "📊", label: "Exporting CSV" },
  export_pdf: { icon: "📃", label: "Exporting PDF" },
};

/* ── Inline PDF / file preview panel ── */
function FilePreview({ url, onClose }) {
  if (!url) return null;
  const downloadUrl = url.includes("?") ? url + "&download=1" : url;
  return (
    <>
      <div className="preview-backdrop" onClick={onClose} />
      <div className="preview-panel">
        <div className="preview-header">
          <span className="preview-title">PDF Preview</span>
          <div className="preview-actions">
            <a href={downloadUrl} download className="preview-dl-btn">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2v9M4 8l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M2 13h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
              Download
            </a>
            <button className="icon-btn preview-close" onClick={onClose}>×</button>
          </div>
        </div>
        <iframe className="preview-iframe" src={url} title="PDF Preview" />
      </div>
    </>
  );
}

/* ── CSV Viewer (inline table) ── */
function CsvViewer({ url, onClose }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!url) return;
    fetch(url).then(r => r.text()).then(text => {
      const lines = text.trim().split("\n").map(l => l.split(",").map(c => c.replace(/^"|"$/g, "").trim()));
      setRows(lines);
    }).catch(e => setError(e.message));
  }, [url]);
  if (!url) return null;
  return (
    <>
      <div className="preview-backdrop" onClick={onClose} />
      <div className="preview-panel">
        <div className="preview-header">
          <span className="preview-title">CSV Preview</span>
          <div className="preview-actions">
            <a href={url} download className="preview-dl-btn">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 2v9M4 8l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M2 13h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
              Download
            </a>
            <button className="icon-btn preview-close" onClick={onClose}>×</button>
          </div>
        </div>
        <div className="csv-table-wrap">
          {error && <p className="csv-error">Failed to load: {error}</p>}
          {rows && (
            <table className="csv-table">
              <thead><tr>{rows[0]?.map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
              <tbody>{rows.slice(1).map((row, ri) => <tr key={ri}>{row.map((c, ci) => <td key={ci}>{c}</td>)}</tr>)}</tbody>
            </table>
          )}
        </div>
      </div>
    </>
  );
}

/* ── Export card (replaces plain links for exported files) ── */
function ExportCard({ href, label, onPreview, onCsvPreview }) {
  const isPdf = href.includes(".pdf");
  const isCsv = href.includes(".csv");
  const fullUrl = href.startsWith("/api/exports/") ? getExportUrl(href) : href;
  return (
    <div className="export-card">
      <div className="export-card-icon">{isPdf ? "📃" : "📊"}</div>
      <div className="export-card-info">
        <span className="export-card-name">{label}</span>
        <span className="export-card-type">{isPdf ? "PDF Document" : "CSV Spreadsheet"}</span>
      </div>
      <div className="export-card-actions">
        {isPdf && (
          <button className="export-btn preview" onClick={() => onPreview(fullUrl)}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M1 7s2.5-5 6-5 6 5 6 5-2.5 5-6 5-6-5-6-5z" stroke="currentColor" strokeWidth="1.3"/><circle cx="7" cy="7" r="2" stroke="currentColor" strokeWidth="1.3"/></svg>
            Preview
          </button>
        )}
        {isCsv && (
          <button className="export-btn preview" onClick={() => onCsvPreview(fullUrl)}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M1 7s2.5-5 6-5 6 5 6 5-2.5 5-6 5-6-5-6-5z" stroke="currentColor" strokeWidth="1.3"/><circle cx="7" cy="7" r="2" stroke="currentColor" strokeWidth="1.3"/></svg>
            View Table
          </button>
        )}
        <a href={fullUrl} download className="export-btn download">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1v8M3.5 6.5L7 10l3.5-3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/><path d="M1.5 12h11" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>
          Download
        </a>
      </div>
    </div>
  );
}

/* ── Inline CSV table from raw text ── */
function InlineCsvTable({ csv }) {
  const rows = csv.trim().split("\n").map(l => l.split(",").map(c => c.replace(/^"|"$/g, "").trim()));
  if (rows.length < 2) return <pre className="code-block">{csv}</pre>;
  return (
    <div className="inline-csv-wrap">
      <table className="csv-table">
        <thead><tr>{rows[0].map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
        <tbody>{rows.slice(1).map((row, ri) => <tr key={ri}>{row.map((c, ci) => <td key={ci}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

/* ── Markdown renderer ── */
function RenderMarkdown({ text, onPreview, onCsvPreview }) {
  if (!text) return null;

  const normalized = text.replace(/\]\s*\n\s*\(/g, "](");

  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
  const segments = [];
  let last = 0;
  let cbMatch;
  while ((cbMatch = codeBlockRegex.exec(normalized)) !== null) {
    if (cbMatch.index > last) segments.push({ type: "md", value: normalized.slice(last, cbMatch.index) });
    const lang = cbMatch[1].toLowerCase();
    const code = cbMatch[2];
    if (lang === "csv" || (!lang && code.includes(",") && code.split("\n").length >= 2 && code.split("\n")[0].split(",").length >= 2)) {
      segments.push({ type: "csv", value: code });
    } else {
      segments.push({ type: "code", lang, value: code });
    }
    last = cbMatch.index + cbMatch[0].length;
  }
  if (last < normalized.length) segments.push({ type: "md", value: normalized.slice(last) });

  return segments.map((seg, si) => {
    if (seg.type === "csv") return <InlineCsvTable key={si} csv={seg.value} />;
    if (seg.type === "code") return <pre key={si} className="code-block">{seg.value}</pre>;
    return <RenderInlineMd key={si} text={seg.value} onPreview={onPreview} onCsvPreview={onCsvPreview} />;
  });
}

function RenderInlineMd({ text, onPreview, onCsvPreview }) {
  const tokens = [];
  const regex = /(\*\*[^*]+\*\*)|(\[([^\]]+)\]\(([^)]+)\))|(\/api\/exports\/[^\s)]+)/g;
  let last = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > last) tokens.push({ type: "text", value: text.slice(last, match.index) });
    if (match[1]) {
      tokens.push({ type: "bold", value: match[1].slice(2, -2) });
    } else if (match[2]) {
      tokens.push({ type: "link", label: match[3], href: match[4] });
    } else if (match[5]) {
      const fname = match[5].split("/").pop();
      tokens.push({ type: "link", label: fname, href: match[5] });
    }
    last = match.index + match[0].length;
  }
  if (last < text.length) tokens.push({ type: "text", value: text.slice(last) });

  return tokens.map((t, i) => {
    if (t.type === "bold") return <strong key={i}>{t.value}</strong>;
    if (t.type === "link") {
      if (t.href.includes("/api/exports/")) {
        return <ExportCard key={i} href={t.href} label={t.label} onPreview={onPreview} onCsvPreview={onCsvPreview} />;
      }
      return <a key={i} href={t.href} target="_blank" rel="noopener noreferrer" className="msg-link">{t.label}</a>;
    }
    return <span key={i}>{t.value}</span>;
  });
}

/* ── Agent reasoning steps ── */
function AgentSteps({ steps }) {
  const [open, setOpen] = useState(true);
  if (!steps || steps.length === 0) return null;
  return (
    <div className="agent-steps">
      <button className="steps-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="steps-arrow" data-open={open}>{"▶"}</span>
        <span className="steps-label">Agent Reasoning</span>
        <span className="steps-count">{steps.length}</span>
      </button>
      {open && (
        <div className="steps-list">
          {steps.map((s, i) => (
            <div key={i} className={`step step-${s.type}`}>
              <div className="step-dot" />
              <div className="step-body">
                {s.type === "thinking" && (
                  <div className="step-row"><span className="step-icon">💡</span><span className="step-text">{s.content}</span></div>
                )}
                {s.type === "tool_start" && (
                  <div className="step-row">
                    <span className="step-icon">{TOOL_LABELS[s.tool]?.icon || "🔧"}</span>
                    <span className="step-text"><strong>{TOOL_LABELS[s.tool]?.label || s.tool}</strong>
                      <span className="step-input">{s.input?.length > 60 ? s.input.slice(0, 60) + "…" : s.input}</span>
                    </span>
                  </div>
                )}
                {s.type === "tool_end" && (
                  <div className="step-result">{s.output?.length > 200 ? s.output.slice(0, 200) + "…" : s.output}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Citations ── */
function Citations({ citations }) {
  const [open, setOpen] = useState(false);
  if (!citations || citations.length === 0) return null;
  const shortName = (src) => { const n = src.replace(/^[a-f0-9-]+_/, ""); return n.length > 28 ? n.slice(0, 26) + "…" : n; };
  return (
    <div className="citations-wrap">
      <button className="citations-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="citations-arrow" data-open={open}>▶</span>
        {citations.length} source{citations.length > 1 ? "s" : ""} cited
      </button>
      {open && (
        <div className="citations-list">
          {citations.map((c, i) => <span key={`${c.source}-${i}`} className="citation-chip">{shortName(c.source)}{c.page != null && ` p.${c.page}`}</span>)}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════ MAIN APP ═══════════════════════ */
export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [statsOpen, setStatsOpen] = useState(false);
  const [sessions, setSessions] = useState([newSession()]);
  const [activeId, setActiveId] = useState(() => sessions[0]?.id);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [ragMode, setRagMode] = useState(true);
  const [selectedModel, setSelectedModel] = useState(modelOptions[0]);
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [ragMetrics, setRagMetrics] = useState([]);
  const [modelUploads, setModelUploads] = useState([]);
  const [selectedModelUploads, setSelectedModelUploads] = useState([]);
  const [notice, setNotice] = useState("");
  const [uploadJob, setUploadJob] = useState(null);
  const [showRagMetrics, setShowRagMetrics] = useState(true);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [csvPreviewUrl, setCsvPreviewUrl] = useState(null);
  const ragInputRef = useRef(null);
  const modelInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const activeSession = sessions.find((s) => s.id === activeId) || sessions[0];

  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); localStorage.setItem("theme", theme); }, [theme]);

  const totals = useMemo(() => {
    const totalDocs = ragMetrics.length;
    const totalTokens = ragMetrics.reduce((a, b) => a + (b.estimated_tokens || 0), 0);
    const totalChunks = ragMetrics.reduce((a, b) => a + (b.chunk_count || 0), 0);
    const indexedChunks = ragMetrics.reduce((a, b) => a + (b.indexed_chunk_count || 0), 0);
    return { totalDocs, totalTokens, totalChunks, indexedChunks };
  }, [ragMetrics]);

  useEffect(() => { refreshRagMetrics(); refreshModelUploads(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [activeSession?.messages]);

  async function refreshRagMetrics() {
    try { const src = await getSources(); const m = await Promise.all(src.map(async (s) => { try { return await getDocumentMetrics(s.doc_id); } catch { return null; } })); setRagMetrics(m.filter(Boolean)); } catch {}
  }
  async function refreshModelUploads() { try { setModelUploads(await getModelUploads()); } catch {} }

  const updateLastBot = useCallback((sessionId, updater) => {
    setSessions((prev) => prev.map((s) => {
      if (s.id !== sessionId) return s;
      const msgs = [...s.messages];
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === "assistant") { const old = msgs[i]; msgs[i] = typeof updater === "function" ? updater(old) : { ...old, ...updater }; break; }
      }
      return { ...s, messages: msgs };
    }));
  }, []);

  function pushMessage(sessionId, msg) {
    setSessions((prev) => prev.map((s) => {
      if (s.id !== sessionId) return s;
      const next = [...s.messages, msg];
      const title = s.title === "New chat" && next.length ? String(next[0].content).slice(0, 40) : s.title;
      return { ...s, messages: next, title };
    }));
  }

  async function handleSend() {
    const q = input.trim(); if (!q || isStreaming) return;
    setInput(""); setIsStreaming(true); setNotice("");
    const sid = activeSession.id;
    pushMessage(sid, { role: "user", content: q });
    pushMessage(sid, { role: "assistant", content: "", steps: [], citations: [], exports: [], latencyMs: null, streaming: true });
    try {
      await streamQuery(
        { question: q, rag_mode: ragMode, selected_model: selectedModel, model_upload_ids: selectedModelUploads },
        {
          onStep(data) { updateLastBot(sid, (prev) => ({ ...prev, steps: [...(prev.steps || []), data] })); },
          onToken(data) { updateLastBot(sid, (prev) => ({ ...prev, content: (prev.content || "") + data.content })); },
          onExports(data) { updateLastBot(sid, (prev) => ({ ...prev, exports: [...(prev.exports || []), ...data] })); },
          onDone(data) { updateLastBot(sid, (prev) => ({ ...prev, latencyMs: data.latency_ms, citations: data.citations?.length ? data.citations : prev.citations, streaming: false })); },
          onError(data) { updateLastBot(sid, (prev) => ({ ...prev, content: prev.content || `Error: ${data.message}`, streaming: false })); },
        }
      );
    } catch (err) { updateLastBot(sid, (prev) => ({ ...prev, content: prev.content || `Error: ${err.message}`, streaming: false })); }
    finally { setIsStreaming(false); }
  }

  async function handleUploadToRag(file) {
    if (!file) return;
    setUploadJob({ filename: file.name, progress: 0, state: "uploading", message: "Uploading…" });
    try {
      const { job_id, doc_id } = await uploadToRag(file);
      setUploadJob((p) => ({ ...p, progress: 5, state: "processing", message: "Processing…" }));
      let status = await getIngestStatus(job_id);
      while (status.state === "queued" || status.state === "processing") {
        setUploadJob({ filename: file.name, progress: status.progress || 0, state: status.state, message: status.message });
        await new Promise((r) => setTimeout(r, 500)); status = await getIngestStatus(job_id);
      }
      if (status.state !== "indexed") throw new Error(status.message || "Ingestion failed");
      setUploadJob({ filename: file.name, progress: 100, state: "indexed", message: status.message });
      const metric = await getDocumentMetrics(doc_id);
      setRagMetrics((prev) => [metric, ...prev.filter((x) => x.doc_id !== doc_id)]);
      setTimeout(() => setUploadJob(null), 2000);
    } catch (err) { setUploadJob({ filename: file.name, progress: 0, state: "failed", message: err.message }); setTimeout(() => setUploadJob(null), 4000); }
  }

  async function handleUploadToModel(file) {
    if (!file) return; setNotice(`Uploading "${file.name}" to model…`);
    try { const out = await uploadToModel(file); await refreshModelUploads(); setSelectedModelUploads((prev) => [...new Set([...prev, out.upload_id])]); setNotice(`Attached: ${file.name}`); }
    catch (err) { setNotice(`Failed: ${err.message}`); }
  }

  function toggleModelUpload(id) { setSelectedModelUploads((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])); }
  async function handleDeleteDoc(docId) { try { await deleteDocument(docId); setRagMetrics((prev) => prev.filter((d) => d.doc_id !== docId)); } catch (err) { setNotice(`Delete failed: ${err.message}`); } }

  return (
    <div className="app-root">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? "" : "closed"}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={() => { const s = newSession(); setSessions((p) => [s, ...p]); setActiveId(s.id); }}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            New chat
          </button>
          <button className="icon-btn sidebar-close-btn" onClick={() => setSidebarOpen(false)} title="Close sidebar">←</button>
        </div>
        <div className="history-list">
          {sessions.map((s) => (
            <button key={s.id} className={`history-item ${s.id === activeId ? "active" : ""}`} onClick={() => setActiveId(s.id)}>
              <span className="history-icon">💬</span><span className="history-title">{s.title}</span>
            </button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <div className="sidebar-label">Model</div>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {modelOptions.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && <button className="icon-btn" onClick={() => setSidebarOpen(true)} title="Open sidebar">☰</button>}
            <div className="mode-toggle" data-mode={ragMode ? "rag" : "chat"}>
              <button className={ragMode ? "active" : ""} onClick={() => setRagMode(true)}>
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 3h10M2 7h6M2 11h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                RAG
              </button>
              <button className={!ragMode ? "active" : ""} onClick={() => setRagMode(false)}>
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="5" stroke="currentColor" strokeWidth="1.5"/><path d="M7 4v3l2 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                Chat
              </button>
              <div className="toggle-slider" />
            </div>
          </div>
          <div className="topbar-right">
            <button className="icon-btn" onClick={() => setTheme((t) => t === "dark" ? "light" : "dark")} title="Toggle theme">
              {theme === "dark" ? "☀️" : "🌙"}
            </button>
            <button className="stats-btn" onClick={() => setStatsOpen((s) => !s)} data-active={statsOpen}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="2" y="8" width="3" height="6" rx="1" fill="currentColor"/><rect x="6.5" y="4" width="3" height="10" rx="1" fill="currentColor"/><rect x="11" y="2" width="3" height="12" rx="1" fill="currentColor"/></svg>
              Stats
              {totals.totalDocs > 0 && <span className="stats-badge">{totals.totalDocs}</span>}
            </button>
          </div>
        </header>

        {uploadJob && (
          <div className="upload-bar">
            <div className="upload-bar-info"><span className="upload-bar-name">{uploadJob.filename}</span><span className="upload-bar-pct">{uploadJob.progress}%</span></div>
            <div className="progress-track"><div className="progress-fill" style={{ width: `${uploadJob.progress}%` }} /></div>
            <div className="upload-bar-msg">{uploadJob.message}</div>
          </div>
        )}

        {/* Messages */}
        <div className="messages">
          <div className="messages-inner">
            {activeSession.messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-logo"><div className="empty-logo-inner">{ragMode ? "📚" : "💬"}</div></div>
                <h1>RAG Studio</h1>
                <p>{ragMode ? "Upload a PDF, then ask questions. The agent will search, reason, and cite sources." : "Chat mode — a helpful assistant with web search and file export tools."}</p>
                <div className="capabilities">
                  {ragMode ? (<>
                    <CapCard icon="🔍" title="Document Search" desc="Hybrid semantic + BM25 retrieval" />
                    <CapCard icon="🌐" title="Web Search" desc="DuckDuckGo for live info" />
                    <CapCard icon="📄" title="Doc Lookup" desc="Metadata of indexed files" />
                    <CapCard icon="📥" title="File Export" desc="Generate CSV or PDF files" />
                  </>) : (<>
                    <CapCard icon="🌐" title="Web Search" desc="DuckDuckGo for live info" />
                    <CapCard icon="📥" title="File Export" desc="Generate CSV or PDF files" />
                    <CapCard icon="💬" title="General Chat" desc="Ask anything, no limits" />
                  </>)}
                </div>
              </div>
            )}

            {activeSession.messages.map((m, i) => (
              <div key={i} className={`msg msg-${m.role}`}>
                <div className="msg-avatar">{m.role === "user" ? "U" : "R"}</div>
                <div className="msg-body">
                  {m.role === "assistant" && <AgentSteps steps={m.steps} />}
                  <div className="msg-text">
                    <RenderMarkdown text={m.content} onPreview={setPreviewUrl} onCsvPreview={setCsvPreviewUrl} />
                    {m.streaming && !m.content && <span className="typing-dots"><span /><span /><span /></span>}
                    {m.streaming && m.content && <span className="cursor-blink" />}
                  </div>
                  {m.role === "assistant" && m.exports && m.exports.length > 0 && (
                    <div className="exports-section">
                      {m.exports.map((ex, ei) => (
                        <ExportCard key={ei} href={ex.href} label={ex.label} onPreview={setPreviewUrl} onCsvPreview={setCsvPreviewUrl} />
                      ))}
                    </div>
                  )}
                  {m.role === "assistant" && <Citations citations={m.citations} />}
                  {m.latencyMs != null && <div className="msg-meta">{m.latencyMs}ms</div>}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Composer */}
        <div className="composer">
          <div className="composer-inner">
            {selectedModelUploads.length > 0 && (
              <div className="attached-files">
                {selectedModelUploads.map((id) => { const item = modelUploads.find((x) => x.upload_id === id); return <button key={id} className="file-chip" onClick={() => toggleModelUpload(id)}>{item?.filename || id} ×</button>; })}
              </div>
            )}
            <div className="composer-row">
              <div className="plus-wrap">
                <button className="plus-btn" onClick={() => setShowPlusMenu((s) => !s)} title="Upload files">
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 4v10M4 9h10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
                </button>
                {showPlusMenu && (
                  <div className="plus-menu">
                    <button onClick={() => { setShowPlusMenu(false); ragInputRef.current?.click(); }}>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 3h12M2 7h8M2 11h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                      Upload to RAG<small>Index for retrieval</small>
                    </button>
                    <button onClick={() => { setShowPlusMenu(false); modelInputRef.current?.click(); }}>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="3" y="2" width="10" height="12" rx="2" stroke="currentColor" strokeWidth="1.5"/><path d="M6 6h4M6 9h2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                      Upload to Model<small>Direct context</small>
                    </button>
                  </div>
                )}
              </div>
              <input className="chat-input" value={input} onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                placeholder={ragMode ? "Ask about your documents…" : "Ask anything…"} disabled={isStreaming} />
              <button className="send-btn" onClick={handleSend} disabled={isStreaming || !input.trim()}>
                {isStreaming ? <span className="sending-spinner" /> : <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 15V3M4 8l5-5 5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>}
              </button>
            </div>
            {notice && <div className="notice">{notice}</div>}
            <div className="composer-footer">{ragMode ? "RAG Mode — agent searches your documents with tools" : "Chat Mode — general assistant with web search"}</div>
          </div>
          <input ref={ragInputRef} type="file" accept=".pdf" hidden onChange={(e) => { handleUploadToRag(e.target.files?.[0]); e.target.value = ""; }} />
          <input ref={modelInputRef} type="file" accept=".pdf,.txt,.md" hidden onChange={(e) => { handleUploadToModel(e.target.files?.[0]); e.target.value = ""; }} />
        </div>
      </div>

      {/* Stats Drawer */}
      {statsOpen && <div className="drawer-backdrop" onClick={() => setStatsOpen(false)} />}
      <aside className={`stats-drawer ${statsOpen ? "open" : ""}`}>
        <div className="drawer-header"><span className="drawer-title">Stats</span><button className="icon-btn" onClick={() => setStatsOpen(false)}>×</button></div>
        <button className="section-toggle" onClick={() => setShowRagMetrics((s) => !s)}>
          <span className="section-arrow" data-open={showRagMetrics}>▶</span> RAG Metrics
        </button>
        {showRagMetrics && (<>
          <div className="stat-grid">
            <StatCard title="Documents" value={totals.totalDocs} />
            <StatCard title="Total Tokens" value={totals.totalTokens.toLocaleString()} />
            <StatCard title="Chunks" value={totals.totalChunks.toLocaleString()} />
            <StatCard title="Indexed" value={totals.indexedChunks.toLocaleString()} />
          </div>
          <h4 className="drawer-section">Model Uploads</h4>
          <div className="drawer-list">
            {modelUploads.length === 0 && <p className="drawer-empty">None yet</p>}
            {modelUploads.map((u) => (<label key={u.upload_id} className="drawer-item checkbox-row"><input type="checkbox" checked={selectedModelUploads.includes(u.upload_id)} onChange={() => toggleModelUpload(u.upload_id)} /><span>{u.filename}</span><small>{u.estimated_tokens.toLocaleString()} tok</small></label>))}
          </div>
          <h4 className="drawer-section">Indexed Documents</h4>
          <div className="drawer-list">
            {ragMetrics.length === 0 && <p className="drawer-empty">No documents indexed</p>}
            {ragMetrics.map((d) => (<div className="drawer-item doc-card" key={d.doc_id}><div className="doc-card-top"><strong>{d.filename.replace(/^[a-f0-9-]+_/, "")}</strong><button className="delete-btn" onClick={() => handleDeleteDoc(d.doc_id)} title="Delete">×</button></div><small>pages: {d.page_count} | ocr: {d.pages_ocr_used ?? 0} | tokens: {d.estimated_tokens.toLocaleString()} | chunks: {d.chunk_count}</small></div>))}
          </div>
        </>)}
      </aside>

      {/* PDF Preview */}
      <FilePreview url={previewUrl} onClose={() => setPreviewUrl(null)} />
      {/* CSV Preview */}
      <CsvViewer url={csvPreviewUrl} onClose={() => setCsvPreviewUrl(null)} />
    </div>
  );
}

function StatCard({ title, value }) { return (<div className="stat-card"><span className="stat-label">{title}</span><strong className="stat-value">{value}</strong></div>); }
function CapCard({ icon, title, desc }) { return (<div className="cap-card"><span className="cap-icon">{icon}</span><strong>{title}</strong><span>{desc}</span></div>); }
