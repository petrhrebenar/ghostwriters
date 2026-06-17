# Anotace rozhodnutí Ústavního soudu — návod

## Cíl
Ve složce `decisions_to_annotate/` je sada textových souborů, každý odpovídá jednomu
rozhodnutí Ústavního soudu. V každém souboru je potřeba **označkovat dvě (resp. tři)
části** textu, které jsou klíčové pro analýzu autorství:

| Značka | Co označuje |
|--------|-------------|
| `RATIO` | **Vlastní právní argumentace soudu** (věcné odůvodnění merita věci). |
| `DIS`   | **Odlišné (disentní) stanovisko** — soudce nesouhlasí s *výrokem*. |
| `CON`   | **Souhlasné (konkurující) stanovisko** — soudce souhlasí s výrokem, ale ne s odůvodněním. |

Ostatní části (hlavička, rekapitulace stížnosti, vyjádření účastníků, procesní
část, výrok) se **nyní neoznačují**.

## Jak značkovat
Do textu vložíte HTML-like značky kolem příslušné pasáže — otevírací na začátku,
uzavírací na konci:

```
<RATIO>
… celý úsek vlastního odůvodnění soudu …
</RATIO>
```

```
<DIS>
Odlišné stanovisko soudce Jana Nováka
… celý text disentu …
</DIS>
```

Tytéž značky platí i pro `CON` (`<CON>…</CON>`).

## Pravidla

**RATIO — vlastní odůvodnění soudu**
- **Začátek:** tam, kde soud přechází od procesních předpokladů a vymezení věci
  k **vlastnímu hodnocení** namítaného porušení práv.
- **Konec:** poslední odstavec věcného odůvodnění. Pokud závěrečný oddíl
  („Závěr") **rekapituluje nosné důvody**, patří i ten do `RATIO`.
- **Nepatří sem:** rekapitulace stížnosti, vyjádření účastníků, procesní posouzení,
  samotný výrok a poučení/náklady řízení.
- Pokud je odůvodnění **přerušeno** (např. vsuvka, která tam nepatří), můžete použít
  **více párů** `<RATIO>…</RATIO>`.

**DIS vs. CON — jak rozlišit**
- Nadpis bývá u obou „Odlišné stanovisko soudce/soudkyně …" — rozhoduje **obsah**:
  - `DIS` = autor by hlasoval pro **jiný výrok** (nesouhlasí s výsledkem; „nesouhlasím
    se zamítnutím/vyhověním", navrhuje jiné rozhodnutí).
  - `CON` = autor souhlasí s výrokem, ale **nesouhlasí/doplňuje odůvodnění**
    („souhlasím s výrokem, avšak…", „k odůvodnění").
- Pozor i na variantu **„Částečně odlišné stanovisko"** — posuďte podle obsahu;
  převažuje-li nesouhlas s výrokem → `DIS`.
- **Každé stanovisko jiného soudce = samostatný pár značek.** Nadpis se jménem
  soudce nechte **uvnitř** značky (kvůli identifikaci autora).

## Důležité
- **Neměňte samotný text** — pouze vkládáte značky. Neopravujte překlepy,
  neslučujte ani nerozdělujte odstavce.
- Řádky začínající `#` (hlavička na začátku souboru) **nechte beze změny**.
- Soubor **uložte pod stejným názvem** (stejná přípona `.txt`, kódování UTF-8).
- Značky se nesmí křížit ani vnořovat (žádné `<RATIO>` uvnitř `<DIS>`).

## Příklad
Viz soubor `priklad_anotace.txt` — ukázka, jak vypadá hotová anotace.

## Úplné definice
Tento návod je zkrácený. Úplné definice kategorií jsou v anotačním schématu
`anotacni_schema_US.md`.
