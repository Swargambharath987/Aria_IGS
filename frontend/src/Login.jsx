import { useState } from 'react';
import { login } from './api.js';
import './Login.css';

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim()) return;
    setError('');
    setLoading(true);
    try {
      const data = await login(username.trim());
      onLogin(data);
    } catch (err) {
      setError(err.message || 'Username not found in IGS directory');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-screen">
      <div className="login-card">
        <div className="login-brand">Aria</div>
        <p className="login-sub">AI Research Computing Assistant</p>
        <form className="login-form" onSubmit={handleSubmit}>
          <label className="login-label">IGS Username</label>
          <input
            className="login-input"
            type="text"
            autoComplete="username"
            placeholder="e.g. jbury"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
            autoFocus
          />
          {error && <p className="login-error">{error}</p>}
          <button className="login-btn" type="submit" disabled={loading || !username.trim()}>
            {loading ? 'Verifying…' : 'Sign in'}
          </button>
        </form>
        <p className="login-hint">Enter your IGS username to continue</p>
      </div>
    </div>
  );
}
