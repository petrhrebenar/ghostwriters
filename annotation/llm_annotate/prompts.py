"""Prompt construction for the full 9-category structural annotation.

The model receives the decision text with every line prefixed by a 1-based
line number and must return a COMPLETE segmentation: an ordered list of
segment *boundaries* (``{start_line, label}``), each running until the next
boundary. This boundary representation makes the output a gap-free,
non-overlapping partition by construction. Categories: see
Poznámky/anotacni_schema_US.md.
"""

from __future__ import annotations

from typing import Dict, List

from .common import meta_header_lines, numbered_text

SYSTEM_MSG = (
    "Jste expertní anotátor rozhodnutí Ústavního soudu ČR. Rozčleníte text"
    " rozhodnutí BEZE ZBYTKU na souvislé úseky a každému přiřadíte právě"
    " jednu z devíti strukturních kategorií. Pracujete pečlivě podle"
    " anotačního schématu a vracíte výhradně platný JSON bez jakéhokoli"
    " doprovodného textu."
)

INSTRUCTIONS = """\
# Úkol
Rozdělte následující rozhodnutí Ústavního soudu BEZE ZBYTKU na za sebou
jdoucí úseky a každému přiřaďte právě jednu z těchto devíti kategorií.
Základní jednotkou je (číslovaný) ODSTAVEC — řiďte se jeho PŘEVAŽUJÍCÍ FUNKCÍ
A MLUVČÍM v daném kontextu, NIKOLI nadpisem oddílu, do něhož spadá.

| Značka | Co označuje |
|--------|-------------|
| HEAD  | Záhlaví: identifikace rozhodnutí a návětí (složení senátu, stěžovatel, předmět) až po uvozovací formuli výroku „takto:". |
| REC   | Rekapitulace stížnosti — soud reprodukuje pozici STĚŽOVATELE (návrh, namítaná práva, jeho argumentace, petit; i pozdější replika stěžovatele). |
| PART  | Vyjádření JINÝCH subjektů než stěžovatele (účastníci, vedlejší účastníci, amici), jak je soud podává. |
| PROC  | Procesní předpoklady a průběh řízení před ÚS (včasnost, oprávněnost, zastoupení, přípustnost; procesní kroky). Řeší ZDA projednat, ne JAK rozhodnout. |
| FACT  | Skutkové a procesní pozadí ve VLASTNÍ NEUTRÁLNÍ řeči soudu (popis napadených rozhodnutí, předchozí řízení, vymezení právní otázky). |
| RATIO | Vlastní VĚCNÉ odůvodnění soudu — konfrontace napadeného aktu s ústavními právy, aplikace zásad/testů, nosné důvody výroku. PRIMÁRNÍ CÍL. |
| DISP  | Výrok a jeho formální rámec: hlavní výrok po „takto:" a závěrečné výrokové konstatování, poučení, náklady, místo/datum/podpis. |
| DIS   | Odlišné (disentní) stanovisko soudce, který NESOUHLASÍ S VÝROKEM. Celý oddíl od nadpisu po podpis. |
| CON   | Souhlasné (konkurující) stanovisko soudce, který souhlasí s výrokem, ale ne s odůvodněním. Celý oddíl od nadpisu po podpis. |

# Klíčová pravidla
- MLUVČÍ > NADPIS: odstavec reprodukující pozici stěžovatele → REC; pozici
  jiného subjektu → PART; vlastní řeč soudu → FACT/PROC/RATIO/DISP dle funkce.
  Jediný oddíl „Rekapitulace" tak může střídat REC, FACT, PROC i PART.
- FACT vs. RATIO: FACT jen rekapituluje pozadí a vymezuje otázku; jakmile soud
  přejde k vlastnímu HODNOCENÍ namítaného porušení, začíná RATIO.
- PROC vs. RATIO: PROC = zda lze věc projednat; RATIO = jak je věc po věcné
  stránce posouzena.
- DISP má typicky DVA úseky (hlavní výrok na začátku + závěrečný výrokový blok).
  Rekapituluje-li závěrečný oddíl nosné důvody, ty patří do RATIO, ne DISP.
- DIS vs. CON (rozhoduje OBSAH, ne nadpis „odlišné stanovisko"): DIS = autor by
  hlasoval pro JINÝ výrok; CON = souhlasí s výrokem, brojí jen proti odůvodnění.
  U „Částečně odlišného stanoviska" převažuje-li nesouhlas s výrokem → DIS.
  Každý soudce = samostatný úsek; nadpis se jménem nechte uvnitř úseku.
- Kategorie se mohou OPAKOVAT a střídat (např. RATIO i DISP vícekrát). Některé
  mohou zcela CHYBĚT (typicky DIS, CON; u usnesení i jiné).

# Formát výstupu
Vraťte POUZE JSON tohoto tvaru (žádný další text, žádné markdown ohraničení):

{
  "segments": [
    {"start_line": 1, "label": "HEAD"},
    {"start_line": <int>, "label": "REC"},
    {"start_line": <int>, "label": "..."}
  ]
}

- Uvádíte jen POČÁTEČNÍ řádek každého úseku; úsek běží až po začátek dalšího,
  poslední až do konce dokumentu. Mezery ani překryvy tak nevznikají.
- První úsek MUSÍ mít start_line = 1. Hodnoty start_line musí být OSTŘE ROSTOUCÍ.
- Sousední úseky musí mít RŮZNÉ značky (jinak je spojte do jednoho).
- Pokryjte CELÝ dokument — každý řádek musí spadat právě do jednoho úseku.
- Povolené značky: HEAD, REC, PART, PROC, FACT, RATIO, DISP, DIS, CON.

# Příklad (ilustrační, zkrácený; čísla = řádky)
Vstup:
  1: Nález ... I. senátu ... ze dne ... takto:
  2: Ústavní stížnost se zamítá.
  3: Odůvodnění
  4: 1. Stěžovatel namítá porušení práva na soudní ochranu a navrhuje zrušení.
  5: 2. Krajský soud ve vyjádření navrhl zamítnutí.
  6: 3. Ústavní stížnost byla podána včas oprávněným stěžovatelem.
  7: 4. Napadeným rozsudkem krajský soud potvrdil rozhodnutí I. stupně.
  8: 5. Ústavní soud připomíná, že právo na soudní ochranu ... (vlastní hodnocení)
  9: 6. Z uvedených důvodů soud stížnost zamítl.
 10: Odlišné stanovisko soudce C. D.
 11: Nesouhlasím s výrokem; stížnosti mělo být vyhověno.
Výstup:
  {"segments": [
    {"start_line": 1, "label": "HEAD"},
    {"start_line": 2, "label": "DISP"},
    {"start_line": 3, "label": "REC"},
    {"start_line": 5, "label": "PART"},
    {"start_line": 6, "label": "PROC"},
    {"start_line": 7, "label": "FACT"},
    {"start_line": 8, "label": "RATIO"},
    {"start_line": 10, "label": "DIS"}
  ]}
(Nadpis „Odůvodnění" na řádku 3 přiřaďte následujícímu věcnému úseku podle
jeho funkce — zde tedy k REC.)
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
