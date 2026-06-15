"""Prompt construction for RATIO/DIS/CON line-range annotation.

The model receives the decision text with every line prefixed by a 1-based
line number and must return JSON describing spans as *line ranges*
(``start_line``/``end_line``), never reproduced text. This is robust to messy
formatting and cheap in tokens.
"""

from __future__ import annotations

from typing import Dict, List

from .common import meta_header_lines, numbered_text

SYSTEM_MSG = (
    "Jste expertní anotátor rozhodnutí Ústavního soudu ČR. Vaším úkolem je "
    "v textu rozhodnutí přesně vymezit pasáže RATIO, DIS a CON pomocí čísel "
    "řádků. Pracujete pečlivě a vracíte výhradně platný JSON bez jakéhokoli "
    "doprovodného textu."
)

INSTRUCTIONS = """\
# Úkol
V následujícím rozhodnutí Ústavního soudu identifikujte pasáže těchto tří typů
a vraťte je jako rozsahy řádků (čísla na začátku každého řádku):

| Značka | Co označuje |
|--------|-------------|
| RATIO | Vlastní právní argumentace soudu (věcné odůvodnění merita věci). |
| DIS   | Odlišné (disentní) stanovisko — soudce nesouhlasí s *výrokem*. |
| CON   | Souhlasné (konkurující) stanovisko — soudce souhlasí s výrokem, ale ne s odůvodněním. |

Ostatní části (hlavička, rekapitulace stížnosti, vyjádření účastníků, procesní
část, výrok, poučení/náklady) se NEOZNAČUJÍ.

# Pravidla
RATIO — vlastní odůvodnění soudu:
- Začátek: tam, kde soud přechází od procesních předpokladů a vymezení věci
  k vlastnímu hodnocení namítaného porušení práv.
- Konec: poslední odstavec věcného odůvodnění. Pokud závěrečný oddíl
  rekapituluje nosné důvody, patří i ten do RATIO.
- Nepatří sem: rekapitulace stížnosti, vyjádření účastníků, procesní posouzení,
  samotný výrok, poučení a náklady řízení.
- Je-li odůvodnění přerušeno vsuvkou, která tam nepatří, použijte VÍCE rozsahů
  RATIO.

DIS vs. CON — jak rozlišit (rozhoduje OBSAH, ne nadpis):
- DIS = autor by hlasoval pro jiný výrok (nesouhlasí s výsledkem).
- CON = autor souhlasí s výrokem, ale nesouhlasí/doplňuje odůvodnění.
- U varianty „Částečně odlišné stanovisko" rozhodněte podle obsahu; převažuje-li
  nesouhlas s výrokem → DIS.
- Každé stanovisko jiného soudce = samostatný rozsah. Nadpis se jménem soudce
  (např. „Odlišné stanovisko soudce Jana Nováka") nechte UVNITŘ rozsahu (na
  začátku jeho prvního řádku) — autora neurčujte, jen ho ponechte v textu.

# Formát výstupu
Vraťte POUZE JSON tohoto tvaru (žádný další text, žádné markdown ohraničení):

{
  "spans": [
    {"label": "RATIO", "start_line": <int>, "end_line": <int>},
    {"label": "DIS", "start_line": <int>, "end_line": <int>},
    {"label": "CON", "start_line": <int>, "end_line": <int>}
  ]
}

- start_line a end_line jsou včetně (oba řádky patří do rozsahu).
- Rozsahy se nesmí překrývat ani vnořovat.
- Vaším úkolem je POUZE vymezit rozsahy; jméno soudce neurčujte.
- Pokud žádná pasáž daného typu není, prostě ji do seznamu nezahrnujte.

# Příklad (ilustrační, zkrácený)
Vstup:
  5: IV. Posouzení důvodnosti
  6: Ústavní soud připomíná, že právo na soudní ochranu ... (vlastní hodnocení)
  7: Z uvedených důvodů Ústavní soud uzavírá, že ... vyhověl.
  8: Odlišné stanovisko soudce C. D.
  9: Nesouhlasím s výrokem nálezu. Mám za to, že ...
Výstup:
  {"spans": [
    {"label": "RATIO", "start_line": 6, "end_line": 7},
    {"label": "DIS", "start_line": 8, "end_line": 9}
  ]}
"""


def build_user_message(meta: Dict, lines: List[str]) -> str:
    """Assemble the user message: rules + metadata header + numbered text."""
    header = "\n".join(meta_header_lines(meta))
    body = numbered_text(lines)
    return (
        f"{INSTRUCTIONS}\n"
        f"# Rozhodnutí k anotaci\n"
        f"{header}\n\n"
        f"Text (každý řádek má číslo; celkem {len(lines)} řádků):\n"
        f"{body}\n"
    )
