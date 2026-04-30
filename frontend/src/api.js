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
