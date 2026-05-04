from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # SSH — IGS login node
    slurm_ssh_host:    str = ""
    slurm_ssh_key_path: str = "/run/secrets/slurm_key"
    slurm_ssh_user:    str = "aria_service"
    slurm_ssh_timeout: int = 15

    # JWT key — validates user tokens (base64-encoded jwt_hs256.key from the cluster)
    slurm_jwt_key: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
