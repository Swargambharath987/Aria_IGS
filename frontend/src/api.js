const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TOKEN = import.meta.env.VITE_API_TOKEN || 'igs-dev-token';

const headers = () => ({
  'Content-Type': 'application/json',
  Authorization: `Bearer ${TOKEN}`,
});

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

export async function sendMessage(message, sessionId, userId) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ message, session_id: sessionId, user_id: userId }),
  });
  if (!res.ok) throw new Error(res.statusText);
  return await res.json();
}

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
 *   onMeta({ session_id })        — fired first with the resolved session id
 *   onToken(text)                 — fired for each LLM token
 *   onDone({ message_id, latency_ms }) — fired when streaming is complete and DB is persisted
 *   onError(Error)                — fired on network or server error
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
        // SSE events are separated by double newlines
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'meta')  onMeta?.(data);
            if (data.type === 'token') onToken?.(data.text);
            if (data.type === 'done')  onDone?.(data);
            if (data.type === 'error') onError?.(new Error(data.message));
          } catch { /* malformed SSE line — skip */ }
        }
      }
    })
    .catch((err) => onError?.(err));
}
