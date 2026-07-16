from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from enum import StrEnum
from io import BytesIO
from pathlib import Path

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer
from markitdown import MarkItDown


class Parser(StrEnum):
    """Closed set of parsing strategies a media type can be routed to."""

    DOCLING = "docling"  # rich-layout documents (PDF, DOCX, PPTX, images)
    MARKITDOWN = "markitdown"  # structured / tabular / web (CSV, JSON, XLSX, HTML)
    TEXT = "text"  # already textual, decoded as-is


# A file resolves its media type from the extension; a remote payload gets it from
# the gateway. Both then dispatch on the media type alone, via the tables below.
_EXTENSION_MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".txt": "text/plain",
}

# Which parser handles each media type. Overlapping formats are assigned deliberately:
# Docling for rich layout, MarkItDown for structured/tabular/web, TEXT decoded as-is.
# A media type not listed here fails loud at conversion time.
_PARSER_BY_MEDIA_TYPE: dict[str, Parser] = {
    "application/pdf": Parser.DOCLING,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Parser.DOCLING,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": Parser.DOCLING,
    "image/png": Parser.DOCLING,
    "image/jpeg": Parser.DOCLING,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": Parser.MARKITDOWN,
    "text/csv": Parser.MARKITDOWN,
    "application/json": Parser.MARKITDOWN,
    "text/html": Parser.MARKITDOWN,
    "text/markdown": Parser.TEXT,
    "text/plain": Parser.TEXT,
}

# A representative extension per media type, so Docling/MarkItDown can detect the
# format from an in-memory stream name (they key off the extension, not the bytes).
_MEDIA_TYPE_EXTENSION = {mt: ext for ext, mt in _EXTENSION_MEDIA_TYPES.items()}


def media_type_for_path(path: Path) -> str:
    """Resolves a file's media type from its extension (a filesystem-source concern)."""
    suffix = path.suffix.lower()
    if suffix in _EXTENSION_MEDIA_TYPES:
        return _EXTENSION_MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is not None:
        return guessed
    raise ValueError(f"Cannot determine media type for file: {path.name}")


class BaseConverter(ABC):
    """Port for turning a raw payload into text, given its media type."""

    @abstractmethod
    def convert_stream(self, data: bytes, media_type: str, name: str = "document") -> str:
        """Converts an in-memory payload to text; ``name`` only aids format detection."""

    @abstractmethod
    def convert_file(self, path: Path) -> str:
        """Converts a file on disk to text (media type resolved from the path)."""


class MarkdownConverter(BaseConverter):
    """Turns a raw byte payload into markdown, dispatching on its media type."""

    def __init__(self, do_table: bool = True, do_ocr: bool = False) -> None:
        options = PdfPipelineOptions()
        options.do_table_structure = do_table
        options.do_ocr = do_ocr
        self.docling = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        self.markitdown = MarkItDown()

    def convert_stream(self, data: bytes, media_type: str, name: str = "document") -> str:
        match _PARSER_BY_MEDIA_TYPE.get(media_type):
            case Parser.DOCLING:
                ext = _MEDIA_TYPE_EXTENSION.get(media_type, "")
                if not name.lower().endswith(ext):
                    name = f"{name}{ext}"
                result = self.docling.convert(DocumentStream(name=name, stream=BytesIO(data)))
                return result.document.export_to_markdown(included_content_layers={ContentLayer.BODY})
            case Parser.MARKITDOWN:
                result = self.markitdown.convert_stream(
                    BytesIO(data), file_extension=_MEDIA_TYPE_EXTENSION.get(media_type)
                )
                return result.text_content
            case Parser.TEXT:
                return data.decode("utf-8")
            case _:
                raise ValueError(f"Unsupported media type: {media_type}")

    def convert_file(self, path: Path) -> str:
        match _PARSER_BY_MEDIA_TYPE.get(media_type_for_path(path)):
            case Parser.DOCLING:
                result = self.docling.convert(str(path))
                return result.document.export_to_markdown(included_content_layers={ContentLayer.BODY})
            case Parser.MARKITDOWN:
                return self.markitdown.convert(str(path)).text_content
            case Parser.TEXT:
                return path.read_text(encoding="utf-8")
            case _:
                raise ValueError(f"Unsupported media type for file: {path.name}")