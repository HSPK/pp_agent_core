"""Image sniffing for the harness read tool.

Python port of `packages/agent/src/harness/tools/image.ts`.
"""

from __future__ import annotations

import base64

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
_BMP_BITS_PER_PIXEL = (1, 4, 8, 16, 24, 32)


def detect_supported_image_mime_type(buffer: bytes) -> str | None:
    """Return the image MIME type sniffed from `buffer`, or `None` if unsupported."""
    if buffer.startswith(bytes([0xFF, 0xD8, 0xFF])):
        return None if len(buffer) > 3 and buffer[3] == 0xF7 else "image/jpeg"
    if buffer.startswith(PNG_SIGNATURE):
        return "image/png" if _is_png(buffer) and not _is_animated_png(buffer) else None
    if _starts_with_ascii(buffer, 0, "GIF"):
        return "image/gif"
    if _starts_with_ascii(buffer, 0, "RIFF") and _starts_with_ascii(buffer, 8, "WEBP"):
        return "image/webp"
    if _starts_with_ascii(buffer, 0, "BM") and _is_bmp(buffer):
        return "image/bmp"
    return None


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _is_png(buffer: bytes) -> bool:
    return (
        len(buffer) >= 16
        and _read_uint32_be(buffer, len(PNG_SIGNATURE)) == 13
        and _starts_with_ascii(buffer, 12, "IHDR")
    )


def _is_animated_png(buffer: bytes) -> bool:
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(buffer):
        chunk_length = _read_uint32_be(buffer, offset)
        chunk_type_offset = offset + 4
        if _starts_with_ascii(buffer, chunk_type_offset, "acTL"):
            return True
        if _starts_with_ascii(buffer, chunk_type_offset, "IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(buffer):
            return False
        offset = next_offset
    return False


def _is_bmp(buffer: bytes) -> bool:
    if len(buffer) < 26:
        return False
    declared_file_size = _read_uint32_le(buffer, 2)
    pixel_data_offset = _read_uint32_le(buffer, 10)
    dib_header_size = _read_uint32_le(buffer, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False

    if dib_header_size == 12:
        color_planes = _read_uint16_le(buffer, 22)
        bits_per_pixel = _read_uint16_le(buffer, 24)
    elif 40 <= dib_header_size <= 124:
        if len(buffer) < 30:
            return False
        color_planes = _read_uint16_le(buffer, 26)
        bits_per_pixel = _read_uint16_le(buffer, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in _BMP_BITS_PER_PIXEL


def _read_uint16_le(buffer: bytes, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 2].ljust(2, b"\x00"), "little")


def _read_uint32_be(buffer: bytes, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 4].ljust(4, b"\x00"), "big")


def _read_uint32_le(buffer: bytes, offset: int) -> int:
    return int.from_bytes(buffer[offset : offset + 4].ljust(4, b"\x00"), "little")


def _starts_with_ascii(buffer: bytes, offset: int, text: str) -> bool:
    if len(buffer) < offset + len(text):
        return False
    return buffer[offset : offset + len(text)] == text.encode("ascii")


__all__ = ["PNG_SIGNATURE", "detect_supported_image_mime_type", "encode_base64"]
