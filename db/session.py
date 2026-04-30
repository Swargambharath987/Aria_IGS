import os
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://aria:aria_dev_password@localhost:5432/aria",
)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,        # connections kept open
    max_overflow=20,     # extra connections allowed under burst
    pool_pre_ping=True,  # test connection before using from pool (handles DB restarts)
    pool_recycle=3600,   # recycle connections after 1h to avoid stale connections
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency — yields a DB session, always closes it."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def db_session():
    """Context manager for use outside of FastAPI request context (scripts, startup)."""
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
