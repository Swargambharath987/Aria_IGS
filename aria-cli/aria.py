#!/usr/bin/env python3
"""
Aria CLI — terminal client for the Aria IGS research computing assistant.

Usage:
    aria ask "how do I submit a GPU job?"
    aria sessions
    aria history <session_id>
    aria status
    aria config
"""
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

import click
import httpx
import yaml

CONFIG_PATH = Path.home() / ".aria" / "config.yml"

DEFAULT_CONFIG = {
    "api_url": "http://localhost:8000",
    "user_id": "dev",
    "token": "igs-dev-token",
    "session_id": None,
}


# ── Config helpers ─────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return {**DEFAULT_CONFIG, **(yaml.safe_load(f) or {})}
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def headers(cfg: dict) -> dict:
    return {"Authorization": f"Bearer {cfg['token']}", "Content-Type": "application/json"}


def _rel_time(ts_str: str) -> str:
    """Return a human-readable relative time string."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        diff = datetime.now(timezone.utc) - ts
        s = int(diff.total_seconds())
        if s < 60:     return "just now"
        if s < 3600:   return f"{s // 60}m ago"
        if s < 86400:  return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except Exception:
        return ts_str


# ── Rich / plain output helpers ────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text
    _console = Console()
    _err     = Console(stderr=True)

    def print_md(text: str):
        _console.print(Markdown(text))

    def print_dim(text: str):
        _console.print(text, style="dim")

    def print_bold(text: str):
        _console.print(text, style="bold")

    def print_err(text: str):
        _err.print(f"[red]{text}[/red]")

except ImportError:
    def print_md(text: str):
        print(text)
    def print_dim(text: str):
        print(text)
    def print_bold(text: str):
        print(text)
    def print_err(text: str):
        print(text, file=sys.stderr)


# ── CLI group ──────────────────────────────────────────────────────────────

@click.group()
@click.option("--url",   default=None, help="Override API URL")
@click.option("--user",  default=None, help="Override user ID")
@click.option("--token", default=None, help="Override bearer token")
@click.pass_context
def cli(ctx, url, user, token):
    """Aria — IGS research computing assistant."""
    cfg = load_config()
    if url:   cfg["api_url"] = url
    if user:  cfg["user_id"] = user
    if token: cfg["token"]   = token
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


# ── aria ask ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--new", is_flag=True, help="Start a new session instead of continuing")
@click.pass_context
def ask(ctx, question, new):
    """Send a message to Aria and stream the response."""
    cfg = ctx.obj["cfg"]
    message = " ".join(question)

    session_id = str(uuid.uuid4()) if new or not cfg.get("session_id") else cfg["session_id"]

    payload = {
        "message":    message,
        "session_id": session_id,
        "user_id":    cfg["user_id"],
    }

    print_dim(f"\nAria  ({cfg['api_url']})\n")

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{cfg['api_url']}/chat/stream",
                json=payload,
                headers={**headers(cfg), "Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode()
                    print_err(f"Error {resp.status_code}: {body}")
                    sys.exit(1)

                full_text = ""
                sources   = []
                message_id = None

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "token":
                        text = data.get("text", "")
                        full_text += text
                        print(text, end="", flush=True)

                    elif data.get("type") == "done":
                        print()  # newline after tokens
                        sources    = data.get("sources") or []
                        message_id = data.get("message_id")

                    elif data.get("type") == "error":
                        print()
                        print_err(f"\nError: {data.get('detail', 'unknown error')}")
                        sys.exit(1)

                if sources:
                    labels = list(dict.fromkeys(s.get("label", "") for s in sources if s.get("label")))
                    if labels:
                        print_dim(f"\nSources: {', '.join(labels)}")

                if message_id:
                    print_dim(f"Message ID: {message_id}  |  Session: {session_id}")

                # Persist session for next `aria ask` call
                cfg["session_id"] = session_id
                save_config(cfg)

    except httpx.ConnectError:
        print_err(f"Cannot connect to Aria at {cfg['api_url']} — is the server running?")
        sys.exit(1)
    except httpx.TimeoutException:
        print_err("Request timed out — the model may still be loading.")
        sys.exit(1)


# ── aria sessions ─────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def sessions(ctx):
    """List your past conversations."""
    cfg = ctx.obj["cfg"]
    try:
        resp = httpx.get(
            f"{cfg['api_url']}/sessions",
            params={"user_id": cfg["user_id"]},
            headers=headers(cfg),
            timeout=15.0,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        print_err(f"Cannot connect to Aria at {cfg['api_url']}")
        sys.exit(1)

    data = resp.json()
    if not data:
        print_dim("No sessions yet. Start one with: aria ask \"hello\"")
        return

    print_bold(f"\n  {'#':<4} {'Title':<50} {'Msgs':>5}  {'Last active'}")
    print_dim("  " + "─" * 75)
    for i, s in enumerate(data, 1):
        title   = (s.get("title") or "Untitled")[:48]
        count   = s.get("message_count", 0)
        last    = _rel_time(s.get("last_active") or s.get("created_at", ""))
        sid     = s.get("session_id", "")
        marker  = "→ " if sid == cfg.get("session_id") else "  "
        print(f"{marker} {i:<4} {title:<50} {count:>5}  {last}  [{sid[:8]}]")
    print()


# ── aria history ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("session_id")
@click.pass_context
def history(ctx, session_id):
    """Show the full message history for a session."""
    cfg = ctx.obj["cfg"]
    try:
        resp = httpx.get(
            f"{cfg['api_url']}/sessions/{session_id}/messages",
            headers=headers(cfg),
            timeout=15.0,
        )
        resp.raise_for_status()
    except httpx.ConnectError:
        print_err(f"Cannot connect to Aria at {cfg['api_url']}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print_err(f"Error: {e.response.status_code}")
        sys.exit(1)

    messages = resp.json()
    if not messages:
        print_dim("No messages in this session.")
        return

    print()
    for msg in messages:
        role    = msg.get("role", "unknown")
        content = msg.get("content", "")
        ts      = _rel_time(msg.get("created_at", ""))

        if role == "user":
            print_bold(f"You  [{ts}]")
            print(f"  {content}\n")
        elif role == "assistant":
            print_bold(f"Aria [{ts}]")
            print_md(content)
            sources = msg.get("sources_used") or []
            if sources:
                labels = list(dict.fromkeys(s.get("label", "") for s in sources if s.get("label")))
                if labels:
                    print_dim(f"  Sources: {', '.join(labels)}")
            print()


# ── aria status ───────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def status(ctx):
    """Check Aria server health."""
    cfg = ctx.obj["cfg"]
    try:
        resp = httpx.get(f"{cfg['api_url']}/health", headers=headers(cfg), timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        print_err(f"Cannot connect to Aria at {cfg['api_url']}")
        sys.exit(1)
    except Exception as e:
        print_err(f"Error: {e}")
        sys.exit(1)

    status_str = data.get("status", "unknown")
    ok = status_str == "ok"
    icon = "✓" if ok else "✗"
    print(f"\n  {icon}  Aria is {'online' if ok else 'degraded'} at {cfg['api_url']}")
    print(f"     Model:   {data.get('model', '—')}")
    print(f"     Chunks:  {data.get('chunks_in_db', '—')}")
    print(f"     LLM:     {'reachable' if data.get('llm_ok') else 'unreachable'}")
    print(f"     DB:      {'reachable' if data.get('db_ok') else 'unreachable'}")
    print()


# ── aria config ───────────────────────────────────────────────────────────

@cli.command("config")
@click.pass_context
def show_config(ctx):
    """Show current configuration."""
    cfg = ctx.obj["cfg"]
    token = cfg.get("token", "")
    masked = token[:6] + "…" + token[-4:] if len(token) > 10 else "***"
    print(f"\n  API URL:    {cfg.get('api_url')}")
    print(f"  User ID:    {cfg.get('user_id')}")
    print(f"  Token:      {masked}")
    print(f"  Session:    {cfg.get('session_id') or '(none — will create on next ask)'}")
    print(f"  Config file: {CONFIG_PATH}\n")


if __name__ == "__main__":
    cli()
