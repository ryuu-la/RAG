import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import Papa from "papaparse";
import { Transformer } from "markmap-lib";
import { Markmap } from "markmap-view";
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

import React from "react";

const DEFAULT_MODELS = [
  { id: "gemma-4-31b-it", label: "Gemma 4 31B", provider: "google" },
  { id: "gemini-3.1-flash-lite-preview", label: "Gemini 3.1 Flash Lite", provider: "google", grounding: true },
  { id: "openai/gpt-oss-120b:free", label: "GPT OSS 120B (Free)", provider: "openrouter" },
];

const TOOL_LABELS = {
  search_documents: { icon: "🔍", label: "Searching documents" },
  lookup_document: { icon: "📄", label: "Looking up document" },
  web_search: { icon: "🌐", label: "Searching the web" },
  image_search: { icon: "📷", label: "Searching for images" },
  read_url: { icon: "📖", label: "Reading webpage" },
  export_csv: { icon: "📊", label: "Exporting CSV" },
  export_pdf: { icon: "📑", label: "Exporting PDF" },
  generate_mindmap: { icon: "🧠", label: "Generating mind map" },
};

const markmapTransformer = new Transformer();

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  render() {
    if (this.state.hasError) return <div className="error-boundary"><h3>Error loading component</h3><p>{this.state.error.message}</p></div>;
    return this.props.children;
  }
}

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
      const parsed = Papa.parse(text.trim());
      if (parsed.errors && parsed.errors.length > 0) throw new Error(parsed.errors[0].message);
      setRows(parsed.data);
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
  const parsed = Papa.parse(csv.trim());
  const rows = parsed.data;
  if (rows.length < 2) return <pre className="code-block">{csv}</pre>;
  return (
    <div className="inline-csv-wrap">
      <table className="csv-table">
        <thead><tr>{rows[0] ? rows[0].map((h, i) => <th key={i}>{h}</th>) : null}</tr></thead>
        <tbody>{rows.slice(1).map((row, ri) => <tr key={ri}>{row.map((c, ci) => <td key={ci}>{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function isCsvHeuristic(code) {
  return code.includes(",") && code.split("\n").length >= 2 && code.split("\n")[0].split(",").length >= 2;
}

// Image Lightbox (click-to-zoom overlay)
function ImageLightbox({ src, alt, onClose }) {
  if (!src) return null;
  return (
    <div className="image-lightbox-backdrop" onClick={onClose}>
      <img src={src} alt={alt || "Image"} className="image-lightbox-img" onClick={e => e.stopPropagation()} />
      <button className="image-lightbox-close" onClick={onClose}>✕</button>
    </div>
  );
}

// MindMap Viewer (fullscreen overlay with toggle indicators)
function MindMapViewer({ data, onClose }) {
  const svgRef = useRef(null);
  const mmRef = useRef(null);
  const enhancingRef = useRef(false);

  // Add separate toggle buttons BETWEEN each node pill and its branches
  const enhanceNodes = useCallback(() => {
    const svg = svgRef.current;
    if (!svg || enhancingRef.current) return;
    enhancingRef.current = true;

    // Remove old toggles (both SVG and HTML leftovers)
    svg.querySelectorAll('.mm-toggle-ind').forEach(el => el.remove());
    svg.querySelectorAll('.mm-toggle-btn').forEach(el => el.remove());

    svg.querySelectorAll('.markmap-node').forEach(gNode => {
      const circle = gNode.querySelector('circle');
      if (!circle) return;
      const r = parseFloat(circle.getAttribute('r') || 0);

      // Leaf node (no children) — truncate the line stub in paddingX gap
      if (r < 1.5) {
        const line = gNode.querySelector('line');
        const fo = gNode.querySelector('foreignObject');
        if (line && fo) {
          const foW = parseFloat(fo.getAttribute('width') || 0);
          line.setAttribute('x2', String(foW));
        }
        return;
      }

      const fill = circle.getAttribute('fill') || '';
      const isCollapsed = fill && fill !== 'none' && fill !== 'transparent'
        && fill !== '#fff' && fill !== '#ffffff' && fill !== 'rgb(255, 255, 255)'
        && !fill.includes('255, 255, 255');

      const ns = 'http://www.w3.org/2000/svg';
      const ind = document.createElementNS(ns, 'g');
      ind.classList.add('mm-toggle-ind');
      // pointer-events: none — clicks pass through to the enlarged circle underneath
      ind.style.pointerEvents = 'none';

      // Background rounded square
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', '-12');
      rect.setAttribute('y', '-12');
      rect.setAttribute('width', '24');
      rect.setAttribute('height', '24');
      rect.setAttribute('rx', '6');

      // Arrow text
      const text = document.createElementNS(ns, 'text');
      text.textContent = isCollapsed ? '›' : '‹';
      text.setAttribute('x', '0');
      text.setAttribute('y', '5');
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('font-size', '16');
      text.setAttribute('font-weight', '700');
      text.setAttribute('font-family', 'Inter, sans-serif');

      ind.appendChild(rect);
      ind.appendChild(text);

      // Position AT the circle (connection point) — sits in the paddingX gap
      const cx = parseFloat(circle.getAttribute('cx') || 0);
      const cy = parseFloat(circle.getAttribute('cy') || 0);
      ind.setAttribute('transform', `translate(${cx}, ${cy})`);

      gNode.appendChild(ind);
    });

    enhancingRef.current = false;
  }, []);

  useEffect(() => {
    if (!svgRef.current || !data?.markdown) return;
    svgRef.current.innerHTML = '';
    const { root } = markmapTransformer.transform(data.markdown);

    // Start fully collapsed: only root node visible
    function foldAll(node) {
      if (!node) return;
      if (node.children?.length) {
        node.payload = { ...(node.payload || {}), fold: 1 };
      }
      node.children?.forEach(foldAll);
    }
    foldAll(root);

    const mm = Markmap.create(svgRef.current, {
      autoFit: true,
      duration: 400,
      maxWidth: 260,
      paddingX: 50,
      spacingVertical: 24,
      spacingHorizontal: 100,
      zoom: true,
      pan: true,
    }, root);
    mmRef.current = mm;
    setTimeout(() => { mm.fit(); enhanceNodes(); }, 200);

    // MutationObserver to re-enhance after collapse/expand animations
    let debounceTimer = null;
    const observer = new MutationObserver(() => {
      if (enhancingRef.current) return;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(enhanceNodes, 180);
    });
    observer.observe(svgRef.current, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['fill', 'r', 'transform'],
    });

    return () => {
      observer.disconnect();
      clearTimeout(debounceTimer);
      try { mm.destroy(); } catch {}
    };
  }, [data?.markdown, enhanceNodes]);

  function handleFit() { mmRef.current?.fit(); }

  // Recursively set fold state on all nodes
  function setExpandAll(node, expand) {
    if (!node) return;
    if (node.payload) { node.payload.fold = expand ? 0 : 1; }
    else { node.payload = { fold: expand ? 0 : 1 }; }
    if (node.children) node.children.forEach(c => setExpandAll(c, expand));
  }

  function handleExpandAll() {
    const mm = mmRef.current;
    if (!mm?.state?.data) return;
    setExpandAll(mm.state.data, true);
    mm.renderData();
    setTimeout(() => { mm.fit(); enhanceNodes(); }, 300);
  }

  function handleCollapseAll() {
    const mm = mmRef.current;
    if (!mm?.state?.data) return;
    // Keep root expanded, collapse everything else
    const root = mm.state.data;
    if (root.payload) root.payload.fold = 0;
    else root.payload = { fold: 0 };
    if (root.children) root.children.forEach(c => setExpandAll(c, false));
    mm.renderData();
    setTimeout(() => { mm.fit(); enhanceNodes(); }, 300);
  }

  function handleDownloadSvg() {
    if (!svgRef.current) return;
    const svgEl = svgRef.current;
    const clone = svgEl.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const styleEl = document.createElement('style');
    styleEl.textContent = `
      svg { background: #1a1b21; }
      .markmap-node-circle { fill-opacity: 0; stroke-opacity: 0; }
      .markmap-node-text { fill: #f0f0f0; font-family: Inter, sans-serif; font-size: 14px; }
      .markmap-link { stroke: #4b5563; fill: none; stroke-width: 2; }
      .markmap-foreign > div { display: inline-block; width: auto; background: rgba(55,65,81,0.9); color: #e2e8f0; padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); font-family: Inter, sans-serif; font-size: 13px; white-space: nowrap; }
      .mm-toggle-ind rect { fill: rgba(55,65,81,0.95); stroke: rgba(255,255,255,0.15); stroke-width: 1; rx: 6; }
      .mm-toggle-ind text { fill: #9ca3af; font-family: Inter, sans-serif; font-size: 16px; font-weight: 700; }
    `;
    clone.insertBefore(styleEl, clone.firstChild);
    const blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(data?.title || 'mindmap').replace(/\s+/g, '_')}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (!data) return null;
  return (
    <>
      <div className="mindmap-backdrop" onClick={onClose} />
      <div className="mindmap-panel">
        <div className="mindmap-header">
          <div className="mindmap-header-info">
            <h2 className="mindmap-title">{data.title}</h2>
            <span className="mindmap-subtitle">Based on {data.source_count || 0} sections &bull; {data.node_count || 0} nodes</span>
          </div>
          <div className="mindmap-actions">
            <button className="mindmap-action-btn" onClick={handleExpandAll} title="Expand all nodes">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4"/></svg>
            </button>
            <button className="mindmap-action-btn" onClick={handleCollapseAll} title="Collapse all nodes">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 14h6v6M14 4h6v6M20 10V4h-6M4 14v6h6"/></svg>
            </button>
            <button className="mindmap-action-btn" onClick={handleFit} title="Fit to screen">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
            </button>
            <button className="mindmap-action-btn" onClick={handleDownloadSvg} title="Download SVG">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
            </button>
            <button className="mindmap-action-btn mindmap-close-btn" onClick={onClose} title="Close">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
            </button>
          </div>
        </div>
        <div className="mindmap-body">
          <svg ref={svgRef} className="mindmap-svg" />
        </div>
        <div className="mindmap-zoom-controls">
          <button className="mindmap-zoom-btn" onClick={handleExpandAll} title="Expand all">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 8V4h4M20 8V4h-4M4 16v4h4M20 16v4h-4"/></svg>
          </button>
          <button className="mindmap-zoom-btn" onClick={handleCollapseAll} title="Collapse all">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 14h6v6M14 4h6v6M20 10V4h-6M4 14v6h6"/></svg>
          </button>
          <button className="mindmap-zoom-btn" onClick={handleFit} title="Fit to screen">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
          </button>
        </div>
      </div>
    </>
  );
}

// MindMap Card (inline in chat message)
function MindMapCard({ data, onOpen }) {
  if (!data) return null;
  return (
    <div className="mindmap-card" onClick={onOpen}>
      <div className="mindmap-card-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="12" cy="12" r="3" />
          <circle cx="4" cy="6" r="2" />
          <circle cx="20" cy="6" r="2" />
          <circle cx="4" cy="18" r="2" />
          <circle cx="20" cy="18" r="2" />
          <path d="M9.5 10.5L5.5 7.5M14.5 10.5L18.5 7.5M9.5 13.5L5.5 16.5M14.5 13.5L18.5 16.5" />
        </svg>
      </div>
      <div className="mindmap-card-info">
        <span className="mindmap-card-title">{data.title}</span>
        <span className="mindmap-card-meta">{data.source_count || 0} sections &bull; {data.node_count || 0} nodes &bull; Interactive Mind Map</span>
      </div>
      <div className="mindmap-card-action">
        <span>View</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
      </div>
    </div>
  );
}

// Markdown renderer
function RenderMarkdown({ text, onPreview, onCsvPreview }) {
  const [lightboxSrc, setLightboxSrc] = useState(null);
  if (!text) return null;
  const normalized = text.replace(/\]\s*\n\s*\(/g, "](");

  return (
    <div className="markdown-body">
      {lightboxSrc && <ImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />}
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          img({ node, src, alt, ...props }) {
            return (
              <img
                src={src}
                alt={alt || "Image"}
                className="chat-thumb"
                onClick={() => setLightboxSrc(src)}
                loading="lazy"
              />
            );
          },
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
              return (
                <div className="code-block-wrapper">
                  <div className="code-block-header">
                    <div className="mac-dots"><span></span><span></span><span></span></div>
                    {lang && <span className="code-lang">{lang}</span>}
                  </div>
                  <pre className="code-block" {...props}><code>{children}</code></pre>
                </div>
              );
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
  const [theme, setTheme] = useState("light");
  const [sessions, setSessions] = useState(() => {
    const saved = localStorage.getItem("sessions");
    return saved ? JSON.parse(saved) : [newSession()];
  });

  useEffect(() => {
    localStorage.setItem("sessions", JSON.stringify(sessions));
  }, [sessions]);
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
  const [mindmapView, setMindmapView] = useState(null);
  const [apiKeyOk, setApiKeyOk] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem("userSettings");
    return saved ? JSON.parse(saved) : { userName: "User Account", temperature: 0.7, topP: 1.0, topK: 40, personality: "" };
  });

  useEffect(() => {
    localStorage.setItem("userSettings", JSON.stringify(settings));
  }, [settings]);

  const ragInputRef = useRef(null);
  const modelInputRef = useRef(null);
  const anyFileInputRef = useRef(null);
  const videoInputRef = useRef(null);
  const imageInputRef = useRef(null);
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

  useEffect(() => {
    refreshModelUploads(); fetchModels(); refreshRagMetrics();
    // Check if API key is configured
    fetch((import.meta.env.VITE_API_BASE || "http://localhost:8000") + "/health")
      .then(r => r.json())
      .then(data => { if (data.api_key_configured === false) setApiKeyOk(false); })
      .catch(() => {});
  }, []);

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
    pushMessage(sid, { role: "assistant", content: "", steps: [], citations: [], exports: [], mindmap: null, latencyMs: null, streaming: true });

    // Reset textarea height
    if (chatInputRef.current) { chatInputRef.current.style.height = 'auto'; }

    try {
      await streamQuery(
        { question: q, rag_mode: ragMode, selected_model: selectedModel, model_upload_ids: selectedModelUploads, history },
        {
          onStep(data) { updateLastBot(sid, (prev) => ({ ...prev, steps: [...(prev.steps || []), data] })); },
          onToken(data) { updateLastBot(sid, (prev) => ({ ...prev, content: (prev.content || "") + data.content })); },
          onExports(data) { updateLastBot(sid, (prev) => ({ ...prev, exports: [...(prev.exports || []), ...data] })); },
          onMindmap(data) { updateLastBot(sid, (prev) => ({ ...prev, mindmap: data })); },
          onDone(data) { updateLastBot(sid, (prev) => ({ ...prev, latencyMs: data.latency_ms, citations: data.citations?.length ? data.citations : prev.citations, streaming: false })); },
          onError(data) { updateLastBot(sid, (prev) => ({ ...prev, content: prev.content || `Error: ${data.message}`, streaming: false })); },
        }
      );
    } catch (err) { updateLastBot(sid, (prev) => ({ ...prev, content: prev.content || `Error: ${err.message}`, streaming: false })); }
    finally { setIsStreaming(false); }
  }

  async function handleUploadToRag(file) {
    if (!file) return;
    if (file.size > 50 * 1024 * 1024) { alert("File too large (max 50MB)"); return; }
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
    if (!file) return; 
    if (file.size > 50 * 1024 * 1024) { alert("File too large (max 50MB)"); return; }
    setNotice(`Uploading "${file.name}" to model...`);
    try { const out = await uploadToModel(file); await refreshModelUploads(); setSelectedModelUploads((prev) => [...new Set([...prev, out.upload_id])]); setNotice(`Attached: ${file.name}`); }
    catch (err) { setNotice(`Failed: ${err.message}`); }
  }

  async function handleDeleteDocument(doc_id, e) {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this document?")) return;
    try {
      await deleteDocument(doc_id);
      refreshRagMetrics();
      setNotice("Document deleted successfully");
    } catch (err) {
      setNotice(`Failed to delete: ${err.message}`);
    }
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
            <div key={s.id} style={{ display: "flex", alignItems: "center", width: "100%", gap: "4px" }}>
              <button onClick={() => setActiveId(s.id)} className={`chat-history-item ${activeId === s.id ? 'active' : ''}`} style={{ flex: 1 }}>
                {s.title}
              </button>
              {sessions.length > 1 && (
                <button 
                  onClick={() => {
                    const newSessions = sessions.filter(session => session.id !== s.id);
                    setSessions(newSessions);
                    if (activeId === s.id) setActiveId(newSessions[0].id);
                  }}
                  style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: "4px 8px", borderRadius: "var(--radius-sm)" }}
                  title="Delete Chat"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M19 6L18 20a2 2 0 01-2 2H81v6M9 6V4a2 2 0 012-2h2a2 2 0 012 2v2"/></svg>
                </button>
              )}
            </div>
          ))}
        </div>
        <div className="sidebar-bottom">
          <div className="user-profile" onClick={() => setShowSettings(true)}>
            <div className="user-avatar-small">{settings.userName ? settings.userName[0].toUpperCase() : "U"}</div>
            <span>{settings.userName || "User Account"}</span>
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
              <button className="icon-btn" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} title="Toggle Theme" style={{ background: "transparent", border: "none", boxShadow: "none" }}>
                {theme === 'dark' ? '☀️' : '🌙'}
              </button>
            <button className="icon-btn" onClick={() => setStatsOpen(s => !s)} title="Stats & Library" style={{ background: "transparent", border: "none", boxShadow: "none" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20v-6M6 20V10M18 20V4" /></svg>
            </button>
          </div>
        </header>

        {!apiKeyOk && (
          <div style={{
            background: 'linear-gradient(135deg, #ff6b3520, #ff4d4d20)',
            border: '1px solid #ff6b3560',
            borderRadius: '12px',
            padding: '14px 20px',
            margin: '10px 20px 0',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '14px',
            color: 'var(--text-primary)',
          }}>
            <span style={{ fontSize: '20px' }}>⚠️</span>
            <div>
              <strong>API Key Not Configured</strong>
              <div style={{ fontSize: '12px', marginTop: '2px', opacity: 0.8 }}>
                Add <code style={{ background: 'var(--bg-tertiary)', padding: '1px 5px', borderRadius: '4px' }}>GOOGLE_API_KEY=your_key</code> to
                your <code style={{ background: 'var(--bg-tertiary)', padding: '1px 5px', borderRadius: '4px' }}>.env</code> file and restart the backend.
                Get a free key at <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer" style={{ color: '#6C8EFF' }}>aistudio.google.com/apikey</a>
              </div>
            </div>
          </div>
        )}

        <div className="chat-stage">
          {activeSession.messages.length === 0 ? null : (
            <div className="message-list">
              <ErrorBoundary>
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
                        {m.role === "assistant" && m.mindmap && (
                          <MindMapCard data={m.mindmap} onOpen={() => setMindmapView(m.mindmap)} />
                        )}
                        {m.role === "assistant" && <Citations citations={m.citations} />}
                      </div>
                    </div>
                  </div>
                ))}
              </ErrorBoundary>
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className={`input-stage ${activeSession.messages.length === 0 ? 'is-empty' : ''}`}>
          <div className="input-box-wrapper" style={{ width: '100%' }}>
            {activeSession.messages.length === 0 && (
              <h2 className="empty-heading" style={{ textAlign: "center", marginBottom: "40px", fontSize: "28px" }}>Where should we begin?</h2>
            )}
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
                <button className="icon-btn" onClick={() => setShowPlusMenu(s => !s)} style={{ background: 'transparent', border: 'none', boxShadow: 'none' }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>
                </button>
                {showPlusMenu && (
                  <div className="attachment-menu">
                    <button onClick={() => { setShowPlusMenu(false); ragInputRef.current?.click(); }}>
                      Upload for RAG (Searchable context)
                    </button>
                    <button onClick={() => { setShowPlusMenu(false); modelInputRef.current?.click(); }}>
                      Upload directly to Model (PDF, TXT)
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
                placeholder={activeSession.messages.length === 0 ? "Ask anything" : "Message AI..."}
                className="chat-textarea"
                rows={1}
                disabled={isStreaming}
                style={{ color: theme === 'dark' ? '#fff' : '#1e1e24' }}
              />
              <button className="send-action-btn" onClick={handleSend} disabled={!input.trim() || isStreaming}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" /></svg>
              </button>
            </div>
            
            {activeSession.messages.length === 0 && (
              <div className="empty-action-chips">
                <button className="action-chip" onClick={() => setInput("Can you summarize the most recent documents?")}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> Summarize documents</button>
                <button className="action-chip" onClick={() => setInput("What are the key insights from the uploaded data?")}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg> Extract insights</button>
              </div>
            )}
          </div>
          <input ref={ragInputRef} type="file" accept=".pdf" hidden onChange={(e) => { handleUploadToRag(e.target.files?.[0]); e.target.value = ""; }} />
          <input ref={modelInputRef} type="file" accept=".pdf,.txt,.md" hidden onChange={(e) => { handleUploadToModel(e.target.files?.[0]); e.target.value = ""; }} />
          <input ref={anyFileInputRef} type="file" hidden onChange={(e) => { handleUploadToModel(e.target.files?.[0]); e.target.value = ""; }} />
          <input ref={videoInputRef} type="file" accept="video/*" hidden onChange={(e) => { handleUploadToModel(e.target.files?.[0]); e.target.value = ""; }} />
          <input ref={imageInputRef} type="file" accept="image/*" hidden onChange={(e) => { handleUploadToModel(e.target.files?.[0]); e.target.value = ""; }} />
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
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="doc-name">{d.filename.replace(/^[a-f0-9-]+_/, "")}</span>
                    <button onClick={(e) => handleDeleteDocument(d.doc_id, e)} className="icon-btn" title="Delete Document" style={{ padding: '2px 4px', fontSize: '12px' }}>🗑️</button>
                  </div>
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
      <MindMapViewer data={mindmapView} onClose={() => setMindmapView(null)} />
      
      {showSettings && (
        <>
          <div className="preview-backdrop" onClick={() => setShowSettings(false)} />
          <div className="preview-panel settings-panel" style={{ maxWidth: '500px', height: 'auto', margin: 'auto', display: 'flex', flexDirection: 'column' }}>
            <div className="preview-header">
              <span className="preview-title">User Settings</span>
              <button className="icon-btn preview-close" onClick={() => setShowSettings(false)}>x</button>
            </div>
            <div style={{ padding: '24px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>User Name</label>
                <input type="text" value={settings.userName} onChange={e => setSettings(s => ({...s, userName: e.target.value}))} style={{ width: '100%', padding: '10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-light)', background: 'var(--bg-input)', color: 'var(--text-primary)' }} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>Temperature ({settings.temperature})</label>
                  <input type="range" min="0" max="2" step="0.1" value={settings.temperature} onChange={e => setSettings(s => ({...s, temperature: parseFloat(e.target.value)}))} style={{ width: '100%' }} />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>Top P ({settings.topP})</label>
                  <input type="range" min="0" max="1" step="0.05" value={settings.topP} onChange={e => setSettings(s => ({...s, topP: parseFloat(e.target.value)}))} style={{ width: '100%' }} />
                </div>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>Top K ({settings.topK})</label>
                <input type="number" min="1" max="100" value={settings.topK} onChange={e => setSettings(s => ({...s, topK: parseInt(e.target.value) || 40}))} style={{ width: '100%', padding: '10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-light)', background: 'var(--bg-input)', color: 'var(--text-primary)' }} />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 600, marginBottom: '6px' }}>Model Personality (System Prompt)</label>
                <textarea rows={4} value={settings.personality} onChange={e => setSettings(s => ({...s, personality: e.target.value}))} placeholder="You are a helpful assistant..." style={{ width: '100%', padding: '10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-light)', background: 'var(--bg-input)', color: 'var(--text-primary)', resize: 'vertical' }} />
              </div>
              <button 
                onClick={() => setShowSettings(false)} 
                style={{ marginTop: '8px', padding: '12px', background: 'var(--accent)', color: '#fff', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 'bold' }}
              >
                Save
              </button>
            </div>
          </div>
        </>
      )}

    </div>
  );
}