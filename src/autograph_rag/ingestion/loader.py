from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer
from markitdown import MarkItDown

from autograph_rag.types import Document, Source, content_hash


class BaseLoader(ABC):
    """Reads raw files from disk and returns a list of Documents."""

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._validate_paths()

    def _validate_paths(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory non trovato: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path non è una directory: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _convert(self, path: Path) -> str:
        """Converts a single file to plain text or markdown."""

    def load_documents(self, save_output: bool = False) -> list[Document]:
        docs: list[Document] = []
        for file_name in os.listdir(self.input_dir):
            path = self.input_dir / file_name
            if not path.is_file():
                continue
            text = self._convert(path)
            source = Source(id=content_hash(text), name=file_name, time=date.today())
            docs.append(Document(text=text, source=source))
            if save_output:
                self._save_output(file_name, text)
        return docs

    def _save_output(self, filename: str, text: str) -> None:
        output_path = self.output_dir / f"{filename}.md"
        output_path.write_text(text, encoding="utf-8")


class DoclingLoader(BaseLoader):
    """Converts documents to markdown using Docling (PDF, DOCX, PPTX, XLSX, HTML, images)."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        do_table: bool = True,
        do_ocr: bool = False,

    ) -> None:
        super().__init__(input_dir, output_dir)
        pdf_options = PdfPipelineOptions()
        pdf_options.do_table_structure = do_table
        pdf_options.do_ocr = do_ocr
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
        )

    def _convert(self, path: Path) -> str:
        result = self.converter.convert(str(path))
        return result.document.export_to_markdown(
            included_content_layers={ContentLayer.BODY}
        )


class MarkItDownLoader(BaseLoader):
    """Converts structured data formats (CSV, JSON, Excel, HTML, DOCX, …) to markdown using MarkItDown."""

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        super().__init__(input_dir, output_dir)
        self.converter = MarkItDown()

    def _convert(self, path: Path) -> str:
        path = str(path)
        converted = self.converter.convert(path)
        return converted.text_content
