import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  deleteDocument,
  getDocumentMetrics,
  getExportUrl,
  getIngestStatus,
  getModels,
  getModelUploads,
  getSources,
  streamQuery,
  uploadToModel,
  uploadToRag,
} from "./api";
import './styles.css';

function newSession() {
  return { id: crypto.randomUUID(), title: "New chat", messages: [] };
}

const DEFAULT_MODELS = [
  { id: "gemma-4-31b-it", label: "Gemma 4 31B", provider: "google" },
  { id: "gemini-3.1-flash-lite-preview", label: "Gemini 3.1 Flash Lite", provider: "google", grounding: true },
];

const TOOL_LABELS = {
  search_documents: { icon: "🔍", label: "Searching documents" },
  lookup_document: { icon: "📄", label: "Looking up document" },
  web_search: { icon: "🌐", label: "Searching the web" },
  read_url: { icon: "📖", label: "Reading webpage" },
  export_csv: { icon: "📊", label: "Exporting CSV" },
  export_pdf: { icon: "📑", label: "Exporting PDF" },
};

// Inline PDF / file preview panel
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
              Download
            </a>
            <button className="icon-btn preview-close" onClick={onClose}>x</button>
          </div>
        </div>
        <iframe className="preview-iframe" src={url} title="PDF Preview" />
      </div>
    </>
  );
}

// CSV Viewer (inline table)
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
              Download
            </a>
            <button className="icon-btn preview-close" onClick={onClose}>x</button>
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

// Export card
function ExportCard({ href, label, onPreview, onCsvPreview }) {
  const isPdf = href.includes(".pdf");
  const isCsv = href.includes(".csv");
  const fullUrl = href.startsWith("/api/exports/") ? getExportUrl(href) : href;
  return (
    <div className="export-card">
      <div className="export-card-icon">{isPdf ? "PDF" : "CSV"}</div>
      <div className="export-card-info">
        <span className="export-card-name">{label}</span>
        <span className="export-card-type">{isPdf ? "PDF Document" : "CSV Spreadsheet"}</span>
      </div>
      <div className="export-card-actions">
        {isPdf && <button className="export-btn preview" onClick={() => onPreview(fullUrl)}>Preview</button>}
        {isCsv && <button className="export-btn preview" onClick={() => onCsvPreview(fullUrl)}>View</button>}
        <a href={fullUrl} download className="export-btn download">Download</a>
      </div>
    </div>
  );
}

// Inline CSV table from raw text
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

function isCsvHeuristic(code) {
  return code.includes(",") && code.split("\n").length >= 2 && code.split("\n")[0].split(",").length >= 2;
}

// Markdown renderer
function RenderMarkdown({ text, onPreview, onCsvPreview }) {
  if (!text) return null;
  const normalized = text.replace(/\]\s*\n\s*\(/g, "](");

  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ node, href, children, ...props }) {
            if (href?.includes("/api/exports/")) {
              let label = children;
              if (Array.isArray(children)) label = children.join("");
              return <ExportCard href={href} label={label || "Export"} onPreview={onPreview} onCsvPreview={onCsvPreview} />;
            }
            return <a href={href} target="_blank" rel="noopener noreferrer" className="msg-link" {...props}>{children}</a>;
          },
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const lang = match ? match[1] : "";
            const codeString = String(children).replace(/\n$/, "");
            if (!inline && (lang === "csv" || (!lang && isCsvHeuristic(codeString)))) {
              return <InlineCsvTable csv={codeString} />;
            }
            if (!inline) {
              return <pre className="code-block" {...props}><code>{children}</code></pre>;
            }
            return <code className="code-inline" {...props}>{children}</code>;
          }
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  );
}

// Agent reasoning steps
function AgentSteps({ steps }) {
  const [open, setOpen] = useState(false);
  if (!steps || steps.length === 0) return null;
  return (
    <div className="agent-steps">
      <button className="steps-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="steps-arrow" data-open={open}>{open ? "▼" : "▶"}</span>
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
                  <div className="step-row"><span className="step-text">{s.content}</span></div>
                )}
                {s.type === "tool_start" && (
                  <div className="step-row">
                    <span className="step-icon">{TOOL_LABELS[s.tool]?.icon || "🔧"}</span>
                    <span className="step-text"><strong>{TOOL_LABELS[s.tool]?.label || s.tool}</strong>
                      <span className="step-input">{s.input?.length > 60 ? s.input.slice(0, 60) + "..." : s.input}</span>
                    </span>
                  </div>
                )}
                {s.type === "tool_end" && (
                  <div className="step-result">{s.output?.length > 200 ? s.output.slice(0, 200) + "..." : s.output}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Citations
function Citations({ citations }) {
  const [open, setOpen] = useState(false);
  if (!citations || citations.length === 0) return null;
  const shortName = (src) => { const n = src.replace(/^[a-f0-9-]+_/, ""); return n.length > 28 ? n.slice(0, 26) + "..." : n; };
  return (
    <div className="citations-wrap">
      <button className="citations-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="citations-arrow">{open ? "▼" : "▶"}</span>
        {citations.length} source{citations.length > 1 ? "s" : ""} cited
      </button>
      {open && (
        <div className="citations-list">
          {citations.map((c, i) => {
            const isUrl = c.source?.startsWith("http");
            if (isUrl) {
              return (
                <a key={`${c.source}-${i}`} href={c.source} target="_blank" rel="noopener noreferrer" className="citation-chip" title={c.source}>
                  {shortName(c.source.replace(/^https?:\/\//, ""))}
                </a>
              );
            }
            return (
              <span key={`${c.source}-${i}`} className="citation-chip">
                {shortName(c.source)}{c.page != null && ` p.${c.page}`}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function shortSourceName(src) {
  if (!src) return "Unknown";
  const clean = src.replace(/^[a-f0-9-]+_/, "");
  if (clean.length > 34) return `${clean.slice(0, 32)}...`;
  return clean;
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [statsOpen, setStatsOpen] = useState(false);
  const [theme, setTheme] = useState("dark");
  const [sessions, setSessions] = useState([newSession()]);
  const [activeId, setActiveId] = useState(() => sessions[0]?.id);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [ragMode, setRagMode] = useState(true);
  const [modelOptions, setModelOptions] = useState(DEFAULT_MODELS);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODELS[0].id);
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [modelUploads, setModelUploads] = useState([]);
  const [selectedModelUploads, setSelectedModelUploads] = useState([]);
  const [ragMetrics, setRagMetrics] = useState([]);
  const [notice, setNotice] = useState("");
  const [uploadJob, setUploadJob] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [csvPreviewUrl, setCsvPreviewUrl] = useState(null);

  const ragInputRef = useRef(null);
  const modelInputRef = useRef(null);
  const chatInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const activeSession = sessions.find((s) => s.id === activeId) || sessions[0];

  const totals = useMemo(() => {
    const totalDocs = ragMetrics.length;
    const totalTokens = ragMetrics.reduce((a, b) => a + (b.estimated_tokens || 0), 0);
    const totalChunks = ragMetrics.reduce((a, b) => a + (b.chunk_count || 0), 0);
    const indexedChunks = ragMetrics.reduce((a, b) => a + (b.indexed_chunk_count || 0), 0);
    return { totalDocs, totalTokens, totalChunks, indexedChunks };
  }, [ragMetrics]);

  useEffect(() => { refreshModelUploads(); fetchModels(); refreshRagMetrics(); }, []);

  async function refreshRagMetrics() {
    try { const src = await getSources(); const m = await Promise.all(src.map(async (s) => { try { return await getDocumentMetrics(s.doc_id); } catch { return null; } })); setRagMetrics(m.filter(Boolean)); } catch { }
  }

  async function fetchModels() { try { const m = await getModels(); if (m?.length) { setModelOptions(m); setSelectedModel(m[0].id); } } catch { } }
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [activeSession?.messages]);

  async function refreshModelUploads() { try { setModelUploads(await getModelUploads()); } catch { } }

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

    // Build conversation history from completed messages in this session
    const history = activeSession.messages
      .filter(m => !m.streaming && m.content)
      .map(m => ({ role: m.role, content: m.content }));

    pushMessage(sid, { role: "user", content: q });
    pushMessage(sid, { role: "assistant", content: "", steps: [], citations: [], exports: [], latencyMs: null, streaming: true });

    // Reset textarea height
    if (chatInputRef.current) { chatInputRef.current.style.height = 'auto'; }

    try {
      await streamQuery(
        { question: q, rag_mode: ragMode, selected_model: selectedModel, model_upload_ids: selectedModelUploads, history },
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
    setUploadJob({ filename: file.name, progress: 0, state: "uploading", message: "Uploading..." });
    try {
      const { job_id, doc_id } = await uploadToRag(file);
      setUploadJob((p) => ({ ...p, progress: 5, state: "processing", message: "Processing..." }));
      let status = await getIngestStatus(job_id);
      while (status.state === "queued" || status.state === "processing") {
        setUploadJob({ filename: file.name, progress: status.progress || 0, state: status.state, message: status.message });
        await new Promise((r) => setTimeout(r, 500)); status = await getIngestStatus(job_id);
      }
      if (status.state !== "indexed") throw new Error(status.message || "Ingestion failed");
      setUploadJob({ filename: file.name, progress: 100, state: "indexed", message: status.message });
      refreshRagMetrics();
      setTimeout(() => setUploadJob(null), 2000);
    } catch (err) { setUploadJob({ filename: file.name, progress: 0, state: "failed", message: err.message }); setTimeout(() => setUploadJob(null), 4000); }
  }

  async function handleUploadToModel(file) {
    if (!file) return; setNotice(`Uploading "${file.name}" to model...`);
    try { const out = await uploadToModel(file); await refreshModelUploads(); setSelectedModelUploads((prev) => [...new Set([...prev, out.upload_id])]); setNotice(`Attached: ${file.name}`); }
    catch (err) { setNotice(`Failed: ${err.message}`); }
  }

  function toggleModelUpload(id) { setSelectedModelUploads((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])); }

  return (
    <div className="layout-root" data-theme={theme}>
      <aside className={`layout-sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-top" style={{ display: 'flex', gap: '8px' }}>
          <button className="new-chat-btn" onClick={() => { const s = newSession(); setSessions(p => [s, ...p]); setActiveId(s.id); }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>
            New chat
          </button>
          <button className="icon-btn" onClick={() => setSidebarOpen(false)} title="Close Sidebar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
          </button>
        </div>
        <div className="sidebar-chats">
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px', paddingLeft: '4px' }}>Today</div>
          {sessions.map(s => (
            <button key={s.id} onClick={() => setActiveId(s.id)} className={`chat-history-item ${activeId === s.id ? 'active' : ''}`}>
              {s.title}
            </button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <div className="user-profile">
            <div className="user-avatar-small">U</div>
            <span>User Account</span>
          </div>
        </div>
      </aside>

      <main className="layout-main">
        <header className="main-header">
          <div className="header-left">
            {!sidebarOpen && (
              <button className="icon-btn" onClick={() => setSidebarOpen(true)}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h18M3 6h18M3 18h18" /></svg>
              </button>
            )}
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="model-selector">
              {modelOptions.map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </div>
          <div className="header-right">
            <button className="icon-btn" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} title="Toggle Theme">
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>
            <button className="icon-btn" onClick={() => setStatsOpen(s => !s)} title="Stats & Library">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20v-6M6 20V10M18 20V4" /></svg>
            </button>
          </div>
        </header>

        <div className="chat-stage">
          {activeSession.messages.length === 0 ? (
            <div className="chat-empty-stage">
              <div className="assistant-logo">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" /></svg>
              </div>
              <h2>How can I help you today?</h2>
            </div>
          ) : (
            <div className="message-list">
              {activeSession.messages.map((m, i) => (
                <div key={i} className={`message-row ${m.role}`}>
                  <div className="message-content">
                    <div className="message-avatar">
                      {m.role === 'user' ? 'U' : <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18"><path d="M12 2a10 10 0 0 0-10 10 10 10 0 0 0 10 10 10 10 0 0 0 10-10A10 10 0 0 0 12 2z" /></svg>}
                    </div>
                    <div className="message-payload">
                      {m.role === "assistant" && <AgentSteps steps={m.steps} />}
                      <div className="message-text">
                        <RenderMarkdown text={m.content} onPreview={setPreviewUrl} onCsvPreview={setCsvPreviewUrl} />
                        {m.streaming && !m.content && <span className="typing-dots"><span /><span /><span /></span>}
                        {m.streaming && m.content && <span className="cursor-blink" />}
                      </div>
                      {m.role === "assistant" && <Citations citations={m.citations} />}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-stage">
          <div className="input-box-wrapper">
            {uploadJob && <div className="upload-notice">{uploadJob.filename}: {uploadJob.progress}% ({uploadJob.message})</div>}
            {notice && <div className="system-notice">{notice}</div>}

            {selectedModelUploads.length > 0 && (
              <div className="attached-files">
                {selectedModelUploads.map(id => {
                  const item = modelUploads.find(x => x.upload_id === id);
                  return <span key={id} className="file-badge">{item?.filename || id} <button onClick={() => toggleModelUpload(id)}>x</button></span>
                })}
              </div>
            )}

            <div className="input-row">
              <div className="plus-btn-wrapper">
                <button className="icon-btn" onClick={() => setShowPlusMenu(s => !s)}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>
                </button>
                {showPlusMenu && (
                  <div className="attachment-menu">
                    <button onClick={() => { setShowPlusMenu(false); ragInputRef.current?.click(); }}>
                      Upload for RAG (Searchable context)
                    </button>
                    <button onClick={() => { setShowPlusMenu(false); modelInputRef.current?.click(); }}>
                      Upload directly to Model
                    </button>
                  </div>
                )}
              </div>
              <textarea
                ref={chatInputRef}
                value={input}
                onChange={e => {
                  setInput(e.target.value);
                  e.target.style.height = 'auto';
                  e.target.style.height = `${e.target.scrollHeight}px`;
                }}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                placeholder="Message Assistant..."
                className="chat-textarea"
                rows={1}
                disabled={isStreaming}
              />
              <button className="send-action-btn" onClick={handleSend} disabled={!input.trim() || isStreaming}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" /></svg>
              </button>
            </div>
            <div className="disclaimer">This UI strictly replicates the clean functionality of modern ChatGPT web.</div>
          </div>
          <input ref={ragInputRef} type="file" accept=".pdf" hidden onChange={(e) => { handleUploadToRag(e.target.files?.[0]); e.target.value = ""; }} />
          <input ref={modelInputRef} type="file" accept=".pdf,.txt,.md" hidden onChange={(e) => { handleUploadToModel(e.target.files?.[0]); e.target.value = ""; }} />
        </div>
      </main>

      <aside className={`insights-sidebar ${statsOpen ? 'open' : 'closed'}`}>
        <div className="insights-header">
          <h3>Library & Stats</h3>
          <button className="icon-btn" onClick={() => setStatsOpen(false)}>x</button>
        </div>
        <div className="insights-content">
          <div className="insights-card">
            <h4>Global RAG Metrics</h4>
            <div className="insights-grid">
              <div className="insights-stat"><span>{totals.totalDocs}</span><small>Documents</small></div>
              <div className="insights-stat"><span>{totals.totalChunks.toLocaleString()}</span><small>Chunks</small></div>
              <div className="insights-stat"><span>{totals.indexedChunks.toLocaleString()}</span><small>Indexed</small></div>
              <div className="insights-stat"><span>{totals.totalTokens.toLocaleString()}</span><small>Tokens</small></div>
            </div>
          </div>

          <div className="insights-card">
            <h4>Indexed Documents</h4>
            {ragMetrics.length === 0 && <p className="empty-text">No documents indexed yet.</p>}
            <ul className="doc-list">
              {ragMetrics.map(d => (
                <li key={d.doc_id}>
                  <span className="doc-name">{d.filename.replace(/^[a-f0-9-]+_/, "")}</span>
                  <div className="doc-meta">
                    {d.page_count} pages &bull; {d.estimated_tokens.toLocaleString()} tokens
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </aside>

      <FilePreview url={previewUrl} onClose={() => setPreviewUrl(null)} />
      <CsvViewer url={csvPreviewUrl} onClose={() => setCsvPreviewUrl(null)} />
    </div>
  );
}