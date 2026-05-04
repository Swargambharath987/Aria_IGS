"""
SSH executor for the Slurm MCP server.

Connects as the service account (aria_service) using the mounted SSH key.
All read commands filter by the requesting user's username.
Write commands (sbatch, scancel) require Slurm operator privileges on aria_service —
pending cluster admin approval from Mike/Dustin.
"""
import paramiko

from config import settings


def run(username: str, command: str) -> str:
    """
    SSH into the Slurm login node as the service account and run a command.
    username is passed for logging/context; the SSH connection uses slurm_ssh_user.

    Returns stdout on success, or a descriptive error string on failure — never raises.
    """
    if not settings.slurm_ssh_host:
        return "[MCP SSH not configured — SLURM_SSH_HOST not set in environment]"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs: dict = {
            "hostname": settings.slurm_ssh_host,
            "username": settings.slurm_ssh_user,
            "timeout":  settings.slurm_ssh_timeout,
        }
        if settings.slurm_ssh_key_path:
            connect_kwargs["key_filename"] = settings.slurm_ssh_key_path

        client.connect(**connect_kwargs)
        _, stdout, stderr = client.exec_command(command, timeout=settings.slurm_ssh_timeout)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        return out if out else (err if err else "(command returned no output)")

    except paramiko.AuthenticationException:
        return (
            f"[SSH authentication failed: {settings.slurm_ssh_user}@{settings.slurm_ssh_host}. "
            "Check that the service key is mounted at SLURM_SSH_KEY_PATH.]"
        )
    except paramiko.SSHException as e:
        return f"[SSH error: {e}]"
    except TimeoutError:
        return f"[SSH timed out connecting to {settings.slurm_ssh_host}]"
    except Exception as e:
        return f"[SSH failed: {type(e).__name__}: {e}]"
    finally:
        client.close()
