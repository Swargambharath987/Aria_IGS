import { useState } from 'react';
import { login } from './api.js';
import './Login.css';

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError]       = useState('');
  const [loading, setLoading]   = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password) return;
    setError('');
    setLoading(true);
    try {
      const data = await login(username.trim(), password);
      onLogin(data);
    } catch (err) {
      setError(err.message || 'Login failed');
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
          <label className="login-label">Username</label>
          <input
            className="login-input"
            type="text"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
            autoFocus
          />
          <label className="login-label">Password</label>
          <input
            className="login-input"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
          />
          {error && <p className="login-error">{error}</p>}
          <button className="login-btn" type="submit" disabled={loading || !username.trim() || !password}>
            {loading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
        <p className="login-hint">IGS research computing credentials</p>
      </div>
    </div>
  );
}
