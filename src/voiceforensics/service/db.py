"""Persistence layer (SQLAlchemy 2.0).

Defaults to SQLite (works everywhere, used in tests); set ``VF_DATABASE_URL`` to a
PostgreSQL URL in production. Models: Job (async analysis lifecycle), ApiKey, and
UsageRecord (metering).
"""

from __future__ import annotations

import datetime as _dt
import secrets
from functools import lru_cache

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from voiceforensics.config import Settings, get_settings


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    analysis_type: Mapped[str] = mapped_column(String(20), default="full")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    active: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    usage: Mapped[list[UsageRecord]] = relationship(back_populates="api_key")


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    endpoint: Mapped[str] = mapped_column(String(80))
    analysis_type: Mapped[str] = mapped_column(String(20), default="")
    created_at: Mapped[_dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    api_key: Mapped[ApiKey | None] = relationship(back_populates="usage")


@lru_cache(maxsize=4)
def _engine_for(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, future=True)


def get_engine(settings: Settings | None = None):
    settings = settings or get_settings()
    return _engine_for(settings.database_url)


def get_sessionmaker(settings: Settings | None = None):
    return sessionmaker(bind=get_engine(settings), expire_on_commit=False, future=True)


def init_db(settings: Settings | None = None) -> None:
    Base.metadata.create_all(get_engine(settings))


def create_api_key(name: str = "", settings: Settings | None = None) -> str:
    """Create and persist a new API key, returning the raw key string."""
    init_db(settings)
    raw = "vfk_" + secrets.token_urlsafe(32)
    Session = get_sessionmaker(settings)
    with Session() as session:
        session.add(ApiKey(key=raw, name=name))
        session.commit()
    return raw
