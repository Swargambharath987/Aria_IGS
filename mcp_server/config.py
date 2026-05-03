from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    slurm_api_url:     str = "http://localhost:6820"
    slurm_api_version: str = "v0.0.41"
    slurm_verify_ssl:  bool = True
    slurm_timeout:     int  = 30
    slurm_jwt_key:     str  = ""   # base64-encoded HMAC key from jwt_hs256.key

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
