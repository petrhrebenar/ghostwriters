"""Text preprocessing: cleaning, tokenization, POS tagging via UDPipe."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ufal.udpipe import Model, Pipeline, ProcessingError

# Default UDPipe model path — user must download the Czech PDT model
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "czech-pdt-ud-2.5-191206.udpipe"


@dataclass
class Token:
    """A single token with its linguistic annotations."""
    form: str
    lemma: str
    upos: str  # Universal POS tag
    xpos: str  # Language-specific POS tag
    feats: str


@dataclass
class Sentence:
    """A sentence as a list of tokens."""
    tokens: list[Token] = field(default_factory=list)

    @property
    def forms(self) -> list[str]:
        return [t.form for t in self.tokens]

    @property
    def lemmas(self) -> list[str]:
        return [t.lemma for t in self.tokens]

    @property
    def upos_tags(self) -> list[str]:
        return [t.upos for t in self.tokens]


@dataclass
class Document:
    """A processed document as a list of sentences."""
    doc_id: str
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def all_tokens(self) -> list[Token]:
        return [t for s in self.sentences for t in s.tokens]

    @property
    def all_forms(self) -> list[str]:
        return [t.form for t in self.all_tokens]

    @property
    def all_lemmas(self) -> list[str]:
        return [t.lemma for t in self.all_tokens]

    @property
    def all_upos(self) -> list[str]:
        return [t.upos for t in self.all_tokens]


def clean_text(text: str) -> str:
    """Basic text cleaning for dissent opinions."""
    if not isinstance(text, str):
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove residual HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Normalize dashes
    text = re.sub(r"[–—]", "-", text)
    # Remove multiple spaces again
    text = re.sub(r" {2,}", " ", text).strip()
    return text


class UDPipeProcessor:
    """Wrapper around UDPipe for tokenization, POS tagging, and lemmatization."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"UDPipe model not found at {model_path}. "
                f"Download from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131 "
                f"and place in {model_path.parent}/"
            )
        self._model = Model.load(str(model_path))
        if not self._model:
            raise RuntimeError(f"Failed to load UDPipe model from {model_path}")
        self._pipeline = Pipeline(
            self._model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
        )
        self._error = ProcessingError()

    def process(self, text: str, doc_id: str = "") -> Document:
        """Process raw text into a Document with sentences and tokens."""
        processed = self._pipeline.process(text, self._error)
        if self._error.occurred():
            raise RuntimeError(f"UDPipe error: {self._error.message}")
        return self._parse_conllu(processed, doc_id)

    @staticmethod
    def _parse_conllu(conllu: str, doc_id: str) -> Document:
        """Parse CoNLL-U format output into Document structure."""
        doc = Document(doc_id=doc_id)
        current_sentence = Sentence()

        for line in conllu.split("\n"):
            line = line.strip()
            if not line:
                if current_sentence.tokens:
                    doc.sentences.append(current_sentence)
                    current_sentence = Sentence()
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            # Skip multi-word tokens (e.g., "1-2")
            if "-" in parts[0] or "." in parts[0]:
                continue
            token = Token(
                form=parts[1],
                lemma=parts[2],
                upos=parts[3],
                xpos=parts[4],
                feats=parts[5],
            )
            current_sentence.tokens.append(token)

        if current_sentence.tokens:
            doc.sentences.append(current_sentence)

        return doc
