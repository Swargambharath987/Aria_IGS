from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # JWT — same secret as the Aria backend; validates user identity from Aria login
    jwt_secret:    str = "aria-dev-secret-change-in-prod"
    jwt_algorithm: str = "HS256"

    # Slurm timeout for subprocess calls
    slurm_timeout: int = 30

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
