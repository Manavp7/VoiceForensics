"""Pluggable object storage: local filesystem (default) or S3/R2 (optional)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from voiceforensics.config import Settings, get_settings


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalStorage(Storage):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Prevent path traversal escaping the storage root.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"invalid storage key: {key}")
        return p

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class S3Storage(Storage):
    """S3 / Cloudflare R2 backend (requires the ``[s3]`` extra: boto3)."""

    def __init__(self, settings: Settings):
        import boto3  # lazy import; only when configured

        self.bucket = settings.s3_bucket
        if not self.bucket:
            raise ValueError("VF_S3_BUCKET must be set for the s3 storage backend")
        self._client = boto3.client(
            "s3", endpoint_url=settings.s3_endpoint_url, region_name=settings.s3_region
        )

    def put(self, key: str, data: bytes) -> str:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return key

    def get(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


def get_storage(settings: Settings | None = None) -> Storage:
    settings = settings or get_settings()
    if settings.storage_backend == "s3":
        return S3Storage(settings)
    return LocalStorage(settings.storage_dir)
