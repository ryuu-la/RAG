const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8001";

async function parseJson(res) {
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || "Request failed");
  }
  return res.json();
}

export async function uploadToRag(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/ingest/upload`, {
    method: "POST",
    body: fd,
  });
  return parseJson(res);
}

export async function getIngestStatus(jobId) {
  const res = await fetch(`${API_BASE}/api/ingest/status/${jobId}`);
  return parseJson(res);
}

export async function getSources() {
  const res = await fetch(`${API_BASE}/api/sources`);
  return parseJson(res);
}

export async function getDocumentMetrics(docId) {
  const res = await fetch(`${API_BASE}/api/metrics/document/${docId}`);
  return parseJson(res);
}

export async function deleteDocument(docId) {
  const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
    method: "DELETE",
  });
  return parseJson(res);
}

export async function uploadToModel(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/model/upload`, {
    method: "POST",
    body: fd,
  });
  return parseJson(res);
}

export async function getModelUploads() {
  const res = await fetch(`${API_BASE}/api/model/uploads`);
  return parseJson(res);
}

export async function sendQuery(payload) {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJson(res);
}

export async function getQueryMetrics(queryId) {
  const res = await fetch(`${API_BASE}/api/metrics/query/${queryId}`);
  return parseJson(res);
}
