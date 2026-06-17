"""Czech function word frequency features."""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from fingerprint.preprocessing import Document

# Czech function words grouped by category.
# This list is intentionally broad — feature selection can prune later.
CZECH_FUNCTION_WORDS: dict[str, list[str]] = {
    "pronouns_personal": [
        "já", "ty", "on", "ona", "ono", "my", "vy", "oni", "ony",
        "mě", "mi", "mne", "mnou", "tě", "ti", "tebe", "tebou",
        "ho", "mu", "ji", "jí", "je", "jej", "ní", "něj", "něm",
        "jemu", "němu", "jim", "jich", "nich", "nimi", "nás", "vás",
        "nám", "vám", "námi", "vámi",
    ],
    "pronouns_reflexive": ["se", "si", "sebe", "sobě", "sebou"],
    "pronouns_demonstrative": [
        "ten", "ta", "to", "ti", "ty", "tím", "tom", "tomu", "té",
        "tu", "těch", "těm", "těmi", "tohoto", "tomto", "toho",
        "tento", "tato", "toto", "tito", "tyto",
    ],
    "pronouns_relative": [
        "který", "která", "které", "kterého", "kterému", "kterém",
        "kterou", "kterým", "kterých", "kterými",
        "jenž", "jež", "jehož", "jemuž", "jímž", "jimiž", "jíž", "nichž",
        "co", "kdo", "čeho", "čemu", "čím", "koho", "komu",
    ],
    "pronouns_indefinite": [
        "něco", "někdo", "nějaký", "některý", "něčí",
        "jakýkoli", "kdokoli", "cokoli", "jakýkoliv",
        "každý", "všechen", "vše", "všichni", "žádný", "nic", "nikdo",
    ],
    "prepositions": [
        "v", "ve", "na", "do", "z", "ze", "s", "se", "k", "ke",
        "o", "u", "po", "za", "od", "ode", "pro", "při", "nad",
        "pod", "před", "mezi", "bez", "přes", "mimo", "kolem",
        "okolo", "vedle", "kvůli", "vůči", "podle", "podél",
        "skrz", "napříč", "dle",
    ],
    "conjunctions": [
        "a", "i", "ale", "nebo", "ani", "avšak", "však",
        "že", "aby", "protože", "neboť", "jelikož",
        "když", "jestliže", "pokud", "kdyby", "zda", "zdali",
        "ačkoli", "ačkoliv", "přestože", "třebaže", "ať",
        "než", "nežli", "zatímco", "jakmile", "dokud",
        "tedy", "tudíž", "proto", "přesto", "přece",
        "buď", "jak", "takže", "čili", "totiž", "ovšem",
    ],
    "auxiliary_verbs": [
        "být", "je", "jsou", "byl", "byla", "bylo", "byli", "byly",
        "jsem", "jsi", "jsme", "jste", "bude", "budou", "budu",
        "budeš", "budeme", "budete", "bych", "bys", "by", "bychom",
        "byste",
    ],
    "particles": [
        "asi", "jen", "již", "už", "ještě", "také", "též", "rovněž",
        "právě", "pouze", "zejména", "především", "sice", "totiž",
        "přece", "vůbec", "dokonce", "přitom", "vlastně", "ostatně",
        "ovšem", "nicméně", "naopak", "spíše", "případně", "zřejmě",
        "patrně", "zjevně", "nepochybně", "bezpochyby",
    ],
    "other": [
        "to", "tak", "pak", "tam", "zde", "tu", "teď", "nyní",
        "potom", "poté", "dále", "dosud", "doposud",
        "ne", "ano", "nikoliv", "nikoli",
    ],
}

# Flat set for fast lookup
ALL_FUNCTION_WORDS: set[str] = set()
for words in CZECH_FUNCTION_WORDS.values():
    ALL_FUNCTION_WORDS.update(words)

# Ordered list for consistent feature vector indexing
FUNCTION_WORD_LIST: list[str] = sorted(ALL_FUNCTION_WORDS)


def function_word_frequencies(
    doc: Document,
    word_list: Optional[list[str]] = None,
    use_lemmas: bool = True,
) -> np.ndarray:
    """Compute relative frequency vector of function words in a document.

    Parameters
    ----------
    doc : Document
        A preprocessed document with tokens.
    word_list : list[str], optional
        Ordered list of function words. Defaults to FUNCTION_WORD_LIST.
    use_lemmas : bool
        If True, match against lemmas; otherwise match against lowercased forms.

    Returns
    -------
    np.ndarray
        Relative frequencies (counts / total tokens) for each function word.
    """
    if word_list is None:
        word_list = FUNCTION_WORD_LIST

    if use_lemmas:
        tokens = [t.lemma.lower() for t in doc.all_tokens]
    else:
        tokens = [t.form.lower() for t in doc.all_tokens]

    total = len(tokens)
    if total == 0:
        return np.zeros(len(word_list))

    counts = Counter(tokens)
    freqs = np.array([counts.get(w, 0) / total for w in word_list])
    return freqs


def function_word_feature_names(
    word_list: Optional[list[str]] = None,
) -> list[str]:
    """Return feature names corresponding to the frequency vector."""
    if word_list is None:
        word_list = FUNCTION_WORD_LIST
    return [f"fw_{w}" for w in word_list]
