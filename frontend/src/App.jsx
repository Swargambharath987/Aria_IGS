import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { fetchSessions, fetchMessages, submitFeedback, streamMessage } from './api.js';
import './App.css';

const USER_ID = import.meta.env.VITE_USER_ID || 'dev';

function MessageBubble({ msg, ratings, onRate }) {
  const isUser = msg.role === 'user';
  const latency = msg.latency_ms != null ? (msg.latency_ms / 1000).toFixed(2) : null;
  const rated = ratings[msg.message_id];
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
            <button
              className={`feedback-btn ${rated === 1 ? 'active-up' : ''}`}
              title="Helpful"
              onClick={() => onRate(msg.message_id, 1)}
            >
              👍
            </button>
            <button
              className={`feedback-btn ${rated === -1 ? 'active-down' : ''}`}
              title="Not helpful"
              onClick={() => onRate(msg.message_id, -1)}
            >
              👎
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [ratings, setRatings] = useState({});
  const bottomRef = useRef(null);
  // Stable ref for the streaming placeholder id so callbacks don't capture stale closures
  const streamingIdRef = useRef(null);

  const loadSessions = useCallback(async () => {
    const data = await fetchSessions(USER_ID);
    setSessions(data);
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function selectSession(sessionId) {
    setActiveSessionId(sessionId);
    setRatings({});
    const msgs = await fetchMessages(sessionId);
    setMessages(msgs);
  }

  function newChat() {
    setActiveSessionId(null);
    setMessages([]);
    setRatings({});
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

    streamMessage(text, activeSessionId ?? null, USER_ID, {
      onMeta: (data) => {
        if (!activeSessionId) setActiveSessionId(data.session_id);
      },
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
          return [
            ...prev.slice(0, -1),
            { ...last, message_id: data.message_id, latency_ms: data.latency_ms, streaming: false },
          ];
        });
        setLoading(false);
        loadSessions();
      },
      onError: (err) => {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last.message_id !== streamingIdRef.current) return prev;
          return [
            ...prev.slice(0, -1),
            { ...last, content: `Error: ${err.message}`, streaming: false },
          ];
        });
        setLoading(false);
      },
    });
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleRate(messageId, rating) {
    setRatings((prev) => ({ ...prev, [messageId]: rating }));
    await submitFeedback(messageId, USER_ID, rating);
  }

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
              className={`session-item ${s.session_id === activeSessionId ? 'active' : ''}`}
              onClick={() => selectSession(s.session_id)}
            >
              <span className="session-title">{s.title || 'Untitled'}</span>
              <span className="session-count">{s.message_count} msgs</span>
            </li>
          ))}
        </ul>
      </aside>

      <main className="chat-area">
        <header className="chat-header">
          <h1>Aria</h1>
          <p>AI Research Computing Assistant</p>
        </header>

        <div className="messages">
          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <div className="empty-icon">🤖</div>
              <h2>Hello, I'm Aria</h2>
              <p>Ask me anything about HPC, Slurm, genomics pipelines, or research computing.</p>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble
              key={msg.message_id}
              msg={msg}
              ratings={ratings}
              onRate={handleRate}
            />
          ))}
          <div ref={bottomRef} />
        </div>

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
      </main>
    </div>
  );
}
