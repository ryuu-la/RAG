const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

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

export async function getModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  return parseJson(res);
}

export async function getQueryMetrics(queryId) {
  const res = await fetch(`${API_BASE}/api/metrics/query/${queryId}`);
  return parseJson(res);
}

export function getExportUrl(path) {
  if (path.startsWith("/")) return `${API_BASE}${path}`;
  return `${API_BASE}/${path}`;
}

export async function streamQuery(payload, { onStep, onToken, onExports, onDone, onError }) {
  const res = await fetch(`${API_BASE}/api/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const txt = await res.text();
    onError?.({ message: txt || "Stream request failed" });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);

      let eventType = "";
      let eventData = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) eventType = line.slice(7);
        else if (line.startsWith("data: ")) eventData = line.slice(6);
      }
      if (!eventType || !eventData) continue;

      try {
        const data = JSON.parse(eventData);
        switch (eventType) {
          case "step":
            onStep?.(data);
            break;
          case "token":
            onToken?.(data);
            break;
          case "exports":
            onExports?.(data);
            break;
          case "done":
            onDone?.(data);
            break;
          case "error":
            onError?.(data);
            break;
        }
      } catch {
        /* malformed JSON, skip */
      }
    }
  }
}
