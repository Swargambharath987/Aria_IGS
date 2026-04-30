import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  fetchSessions, fetchMessages, submitFeedback, streamMessage,
  fetchAdminStats, fetchAdminDocs, deleteAdminDoc, ingestUrl,
  fetchMe, deleteSession,
} from './api.js';
import LoginPage from './Login.jsx';
import './App.css';

// ── Admin panel ────────────────────────────────────────────────────────────

function AdminPanel({ userId }) {
  const [stats, setStats]       = useState(null);
  const [docs, setDocs]         = useState([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);
  const [urlInput, setUrlInput] = useState('');
  const [urlCol, setUrlCol]     = useState('slurm');
  const [urlStatus, setUrlStatus] = useState(null);
  const [ingesting, setIngesting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [s, d] = await Promise.all([fetchAdminStats(), fetchAdminDocs()]);
      setStats(s); setDocs(d);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleDelete(docId, filename) {
    if (!confirm(`Deactivate "${filename}"? It will no longer be used in RAG retrieval.`)) return;
    await deleteAdminDoc(docId);
    setDocs((prev) => prev.filter((d) => d.doc_id !== docId));
    if (stats) setStats((s) => ({ ...s, total_docs: s.total_docs - 1 }));
  }

  async function handleIngestUrl(e) {
    e.preventDefault();
    if (!urlInput.trim()) return;
    setIngesting(true); setUrlStatus(null);
    try {
      const res = await ingestUrl(urlInput.trim(), urlCol, userId);
      setUrlStatus({ ok: true, msg: `Ingested ${res.chunks_added} chunks from ${res.filename}` });
      setUrlInput('');
      load();
    } catch (err) {
      setUrlStatus({ ok: false, msg: err.message });
    } finally {
      setIngesting(false);
    }
  }

  function fmt(bytes) {
    if (bytes == null) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  if (loading) return <div className="admin-loading">Loading…</div>;
  if (error)   return <div className="admin-error">Error: {error}</div>;

  return (
    <div className="admin-panel">
      <div className="admin-header">
        <h2>Admin</h2>
        <button className="admin-refresh" onClick={load} title="Refresh">↻</button>
      </div>

      {stats && (
        <div className="stat-grid">
          <div className="stat-card"><span className="stat-val">{stats.total_users}</span><span className="stat-label">Users</span></div>
          <div className="stat-card"><span className="stat-val">{stats.total_sessions}</span><span className="stat-label">Sessions</span></div>
          <div className="stat-card"><span className="stat-val">{stats.total_messages}</span><span className="stat-label">Messages</span></div>
          <div className="stat-card"><span className="stat-val">{stats.total_docs}</span><span className="stat-label">Docs</span></div>
          <div className="stat-card"><span className="stat-val">{stats.total_chunks}</span><span className="stat-label">Chunks</span></div>
          <div className="stat-card"><span className="stat-val">{stats.avg_latency_ms != null ? `${(stats.avg_latency_ms / 1000).toFixed(1)}s` : '—'}</span><span className="stat-label">Avg latency</span></div>
          <div className="stat-card"><span className="stat-val" style={{ color: '#4caf50' }}>{stats.feedback_up}</span><span className="stat-label">Thumbs up</span></div>
          <div className="stat-card"><span className="stat-val" style={{ color: '#f44336' }}>{stats.feedback_down}</span><span className="stat-label">Thumbs down</span></div>
        </div>
      )}

      <div className="admin-section">
        <h3>Ingest URL</h3>
        <form className="url-ingest-form" onSubmit={handleIngestUrl}>
          <input
            className="url-input"
            type="url"
            placeholder="https://slurm.schedmd.com/..."
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            disabled={ingesting}
          />
          <select className="url-col-select" value={urlCol} onChange={(e) => setUrlCol(e.target.value)} disabled={ingesting}>
            <option value="slurm">slurm</option>
            <option value="institute">institute</option>
            <option value="general">general</option>
          </select>
          <button className="url-ingest-btn" type="submit" disabled={ingesting || !urlInput.trim()}>
            {ingesting ? 'Ingesting…' : 'Ingest'}
          </button>
        </form>
        {urlStatus && (
          <p className={urlStatus.ok ? 'ingest-ok' : 'ingest-err'}>{urlStatus.msg}</p>
        )}
      </div>

      {stats?.top_collections?.length > 0 && (
        <div className="admin-section">
          <h3>Collections</h3>
          <table className="admin-table">
            <thead><tr><th>Collection</th><th>Docs</th><th>Chunks</th></tr></thead>
            <tbody>
              {stats.top_collections.map((c) => (
                <tr key={c.collection}>
                  <td><span className="collection-badge">{c.collection}</span></td>
                  <td>{c.docs}</td><td>{c.chunks}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="admin-section">
        <h3>Knowledge Base ({docs.length} documents)</h3>
        {docs.length === 0 ? (
          <p className="admin-empty">No documents ingested yet.</p>
        ) : (
          <table className="admin-table">
            <thead><tr><th>File / URL</th><th>Collection</th><th>Chunks</th><th>Size</th><th></th></tr></thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.doc_id}>
                  <td className="doc-name" title={d.filename}>{d.filename}</td>
                  <td><span className="collection-badge">{d.collection}</span></td>
                  <td>{d.chunk_count}</td>
                  <td>{fmt(d.file_size_bytes)}</td>
                  <td>
                    <button className="delete-btn" onClick={() => handleDelete(d.doc_id, d.filename)} title="Deactivate">✕</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ── Message bubble ─────────────────────────────────────────────────────────

function MessageBubble({ msg, ratings, onRate }) {
  const isUser    = msg.role === 'user';
  const latency   = msg.latency_ms != null ? (msg.latency_ms / 1000).toFixed(2) : null;
  const rated     = ratings[msg.message_id];
  const isStreaming = msg.streaming && !msg.content;

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      <div className={`bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
        {isStreaming ? (
          <span className="dots"><span /><span /><span /></span>
        ) : isUser ? (
          <p>{msg.content}</p>
        ) : (
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        )}
        {!isUser && !msg.streaming && (
          <div className="msg-meta">
            {latency && <span className="latency">{latency}s</span>}
            <button className={`feedback-btn ${rated === 1 ? 'active-up' : ''}`} title="Helpful" onClick={() => onRate(msg.message_id, 1)}>👍</button>
            <button className={`feedback-btn ${rated === -1 ? 'active-down' : ''}`} title="Not helpful" onClick={() => onRate(msg.message_id, -1)}>👎</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────

export default function App() {
  const [user, setUser]             = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [sessions, setSessions]     = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState('');
  const [loading, setLoading]       = useState(false);
  const [ratings, setRatings]       = useState({});
  const [adminOpen, setAdminOpen]   = useState(false);
  const bottomRef    = useRef(null);
  const streamingIdRef = useRef(null);

  // Check stored JWT on mount
  useEffect(() => {
    const token = localStorage.getItem('aria_jwt');
    if (token) {
      fetchMe()
        .then((u) => { setUser(u); setAuthChecked(true); })
        .catch(() => { localStorage.removeItem('aria_jwt'); setAuthChecked(true); });
    } else {
      setAuthChecked(true);
    }
  }, []);

  const loadSessions = useCallback(async () => {
    if (!user) return;
    const data = await fetchSessions(user.username);
    setSessions(data);
  }, [user]);

  useEffect(() => { loadSessions(); }, [loadSessions]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  function handleLoginSuccess(data) {
    localStorage.setItem('aria_jwt', data.access_token);
    setUser({ username: data.username, role: data.role });
  }

  function handleLogout() {
    localStorage.removeItem('aria_jwt');
    setUser(null);
    setSessions([]);
    setMessages([]);
    setActiveSessionId(null);
    setAdminOpen(false);
  }

  async function selectSession(sessionId) {
    setActiveSessionId(sessionId);
    setAdminOpen(false);
    setRatings({});
    const msgs = await fetchMessages(sessionId);
    setMessages(msgs);
  }

  async function handleDeleteSession(e, sessionId) {
    e.stopPropagation();
    await deleteSession(sessionId);
    setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
    if (activeSessionId === sessionId) {
      setActiveSessionId(null);
      setMessages([]);
    }
  }

  function newChat() {
    setActiveSessionId(null);
    setMessages([]);
    setRatings({});
    setAdminOpen(false);
  }

  function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { message_id: `tmp-${Date.now()}`, role: 'user', content: text };
    const sid = `streaming-${Date.now()}`;
    streamingIdRef.current = sid;

    setMessages((prev) => [
      ...prev,
      userMsg,
      { message_id: sid, role: 'assistant', content: '', streaming: true },
    ]);
    setInput('');
    setLoading(true);

    streamMessage(text, activeSessionId ?? null, user?.username ?? 'dev', {
      onMeta: (data) => { if (!activeSessionId) setActiveSessionId(data.session_id); },
      onToken: (token) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last.message_id !== streamingIdRef.current) return prev;
          return [...prev.slice(0, -1), { ...last, content: last.content + token }];
        });
      },
      onDone: (data) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last.message_id !== streamingIdRef.current) return prev;
          return [...prev.slice(0, -1), { ...last, message_id: data.message_id, latency_ms: data.latency_ms, streaming: false }];
        });
        setLoading(false);
        loadSessions();
      },
      onError: (err) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last.message_id !== streamingIdRef.current) return prev;
          return [...prev.slice(0, -1), { ...last, content: `Error: ${err.message}`, streaming: false }];
        });
        setLoading(false);
      },
    });
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  async function handleRate(messageId, rating) {
    setRatings((prev) => ({ ...prev, [messageId]: rating }));
    await submitFeedback(messageId, user?.username ?? 'dev', rating);
  }

  if (!authChecked) return null;
  if (!user) return <LoginPage onLogin={handleLoginSuccess} />;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="brand">Aria</span>
          <button className="new-chat-btn" onClick={newChat}>+ New Chat</button>
        </div>
        <ul className="session-list">
          {sessions.map((s) => (
            <li
              key={s.session_id}
              className={`session-item ${s.session_id === activeSessionId && !adminOpen ? 'active' : ''}`}
              onClick={() => selectSession(s.session_id)}
            >
              <span className="session-title">{s.title || 'Untitled'}</span>
              <span className="session-right">
                <span className="session-count">{s.message_count} msgs</span>
                <button
                  className="session-del"
                  title="Delete session"
                  onClick={(e) => handleDeleteSession(e, s.session_id)}
                >✕</button>
              </span>
            </li>
          ))}
        </ul>
        <div className="sidebar-footer">
          {user.role === 'admin' && (
            <button
              className={`admin-toggle-btn ${adminOpen ? 'active' : ''}`}
              onClick={() => setAdminOpen((o) => !o)}
            >
              ⚙ Admin
            </button>
          )}
          <div className="user-row-footer">
            <span className="user-name">{user.username}</span>
            <button className="logout-btn" onClick={handleLogout}>Sign out</button>
          </div>
        </div>
      </aside>

      <main className="chat-area">
        <header className="chat-header">
          <h1>{adminOpen ? 'Admin Panel' : 'Aria'}</h1>
          <p>{adminOpen ? 'Knowledge base & usage stats' : 'AI Research Computing Assistant'}</p>
        </header>

        {adminOpen && <AdminPanel userId={user.username} />}

        <div className="messages" style={{ display: adminOpen ? 'none' : undefined }}>
          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <div className="empty-icon">🤖</div>
              <h2>Hello, I'm Aria</h2>
              <p>Ask me anything about HPC, Slurm, genomics pipelines, or research computing.</p>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble key={msg.message_id} msg={msg} ratings={ratings} onRate={handleRate} />
          ))}
          <div ref={bottomRef} />
        </div>

        {!adminOpen && (
          <div className="input-area">
            <textarea
              className="input-box"
              rows={1}
              placeholder="Ask Aria something… (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
            <button className="send-btn" onClick={handleSend} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
