import hashlib
from datetime import date

from pydantic import BaseModel, Field


def content_hash(text: str, length: int = 16) -> str:
    """Deterministic fingerprint of text content, used as a stable/idempotent id.
    """
    
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


class Source(BaseModel):
    """Identifies the origin document of a chunk."""

    id: str = Field(description="Identificativo stabile del documento, hash del contenuto completo")
    name: str = Field(description="Il nome del file della sorgente dati")
    time: date = Field(description="La data di caricamento della sorgente")


class Document(BaseModel):
    """Raw document as loaded from disk, before chunking."""

    text: str = Field(description="Il testo completo del documento originale")
    source: Source = Field(description="La sorgente del documento")


class Metadata(BaseModel):
    """Contextual metadata attached to a single chunk."""

    source: Source = Field(description="La sorgente di questo specifico frammento")
    title: str = Field(description="Il titolo della sezione del documento da cui è estratto il chunk")
    page: int | None = Field(default=None, description="Numero opzionale di pagina")


class Chunk(BaseModel):
    """Atomic unit of text that flows through the retrieval pipeline."""

    id: str = Field(description="Identificativo univoco del chunk")
    text: str = Field(description="Il testo del frammento informativo")
    metadata: Metadata = Field(description="I metadati del chunk")


class ScoredChunk(BaseModel):
    """Chunk paired with a retrieval or ranking score."""

    chunk: Chunk = Field(description="Il frammento informativo estratto")
    score: float = Field(description="Il punteggio di rilevanza lessicale/semantica")
