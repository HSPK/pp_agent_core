"""JSONL v4 session storage subpackage.

Python port of `packages/agent/src/harness/session/jsonl.ts` (the
`jsonl.ts` re-export barrel; the file-per-module implementation itself lives
under `packages/agent/src/harness/session/jsonl/`, mirrored here 1:1).
"""

from __future__ import annotations

from .codec import (
    decode_entry,
    decode_header,
    decode_mutation,
    decode_provisioned_entry,
    decode_record,
    encode_header,
    encode_mutation,
    entry_to_wire,
    metadata_from_header,
    parse_header,
    parse_mutation,
    provisioned_entry_to_wire,
    record_to_wire,
)
from .errors import JsonlDecodeError, invalid_file
from .lockfile import FileLock, LockTimeoutError
from .repo import JsonlSessionRepo
from .storage import JsonlSessionStorage
from .types import (
    JsonlForkOptions,
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepoOptions,
    JsonlV4Header,
)

__all__ = [
    "FileLock",
    "JsonlDecodeError",
    "JsonlForkOptions",
    "JsonlSessionCreateOptions",
    "JsonlSessionListOptions",
    "JsonlSessionMetadata",
    "JsonlSessionRepo",
    "JsonlSessionRepoOptions",
    "JsonlSessionStorage",
    "JsonlV4Header",
    "LockTimeoutError",
    "decode_entry",
    "decode_header",
    "decode_mutation",
    "decode_provisioned_entry",
    "decode_record",
    "encode_header",
    "encode_mutation",
    "entry_to_wire",
    "invalid_file",
    "metadata_from_header",
    "parse_header",
    "parse_mutation",
    "provisioned_entry_to_wire",
    "record_to_wire",
]
