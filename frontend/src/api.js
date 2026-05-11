const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// JWT from login takes priority; fall back to static dev token for local curl-style usage
const getToken = () =>
  localStorage.getItem('aria_jwt') || import.meta.env.VITE_API_TOKEN || 'igs-dev-token';

const headers = () => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${getToken()}`,
});

// ── Auth ──────────────────────────────────────────────────────────────────

export async function login(username) {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return await res.json(); // { access_token, username, role }
}

export async function fetchMe() {
  const res = await fetch(`${BASE_URL}/auth/me`, { headers: headers() });
  if (!res.ok) throw new Error(res.statusText);
  return await res.json(); // { username, role }
}

// ── Sessions ──────────────────────────────────────────────────────────────

export async function fetchSessions(userId) {
  try {
    const res = await fetch(`${BASE_URL}/sessions?user_id=${userId}`, { headers: headers() });
    if (!res.ok) throw new Error(res.statusText);
    return await res.json();
  } catch {
    return [];
  }
}

export async function fetchMessages(sessionId) {
  try {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}/messages`, { headers: headers() });
    if (!res.ok) throw new Error(res.statusText);
    return await res.json();
  } catch {
    return [];
  }
}

export async function deleteSession(sessionId) {
  const res = await fetch(`${BASE_URL}/session/${sessionId}`, {
    method: 'DELETE',
    headers: headers(),
  });
  if (!res.ok) throw new Error(res.statusText);
  return await res.json();
}

// ── Chat ──────────────────────────────────────────────────────────────────

export async function submitFeedback(messageId, userId, rating, comment = '') {
  try {
    const res = await fetch(`${BASE_URL}/feedback`, {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify({ message_id: messageId, user_id: userId, rating, comment }),
    });
    if (!res.ok) throw new Error(res.statusText);
    return await res.json();
  } catch {
    return { status: 'error' };
  }
}

/**
 * Stream a chat message via SSE. Calls callbacks as events arrive:
 *   onMeta({ session_id })
 *   onToken(text)
 *   onDone({ message_id, latency_ms })
 *   onError(Error)
 */
export function streamMessage(message, sessionId, userId, { onMeta, onToken, onDone, onError }) {
  fetch(`${BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ message, session_id: sessionId, user_id: userId }),
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(res.statusText);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'meta')  onMeta?.(data);
            if (data.type === 'token') onToken?.(data.text);
            if (data.type === 'done')  onDone?.({ message_id: data.message_id, latency_ms: data.latency_ms, sources: data.sources || [], pending_action: data.pending_action || null });
            if (data.type === 'error') onError?.(new Error(data.message));
          } catch { /* malformed SSE line — skip */ }
        }
      }
    })
    .catch((err) => onError?.(err));
}

// ── Action confirm ────────────────────────────────────────────────────────

/**
 * Confirm or deny a pending Slurm action.
 * @param {string} actionId - UUID of the pending action
 * @param {'approve'|'deny'} decision
 * @returns {Promise<{status: string, result?: string, action_id: string}>}
 */
export async function confirmAction(actionId, decision) {
  const res = await fetch(`${BASE_URL}/actions/confirm`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ action_id: actionId, decision }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return await res.json();
}

// ── Admin ─────────────────────────────────────────────────────────────────

export async function fetchAdminStats() {
  const res = await fetch(`${BASE_URL}/admin/stats`, { headers: headers() });
  if (!res.ok) throw new Error(res.statusText);
  return await res.json();
}

export async function fetchAdminDocs(collection = null) {
  const url = collection
    ? `${BASE_URL}/admin/docs?collection=${collection}`
    : `${BASE_URL}/admin/docs`;
  const res = await fetch(url, { headers: headers() });
  if (!res.ok) throw new Error(res.statusText);
  return await res.json();
}

export async function deleteAdminDoc(docId) {
  const res = await fetch(`${BASE_URL}/admin/docs/${docId}`, {
    method: 'DELETE',
    headers: headers(),
  });
  if (!res.ok) throw new Error(res.statusText);
  return await res.json();
}

export async function ingestUrl(url, collection, userId) {
  const res = await fetch(`${BASE_URL}/ingest/url`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ url, collection, user_id: userId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return await res.json();
}
