# CCC Dataset — Variable Reference for Re-implementation

> Based on the [CCC Dataset](https://github.com/stepanpaulik/ccc_dataset) by Štěpán Paulík (codebook v1.0, data 1993–2023).
> This document specifies the variables we want in **our** dataset, how Paulík originally obtained them, and their data types/formats.

Each variable has three fields:

| Field | Meaning |
|---|---|
| **Definition** | What the variable captures |
| **Source / method** | How Paulík obtained it — scraped from NALUS, regex-mined from decision texts, or derived/inferred |
| **Type / format / example values** | Data type and illustrative values |

---

## `doc_id`

| | |
|---|---|
| **Definition** | Unique identifier for each **decision**. Primary key used across all decision-level datasets. |
| **Source / method** | Scraped from NALUS (the CCC's online database). Each decision page on NALUS has a unique document identifier. |
| **Type / format** | `string` — e.g. `"doc_12345"` |

---

## `case_id`

| | |
|---|---|
| **Definition** | Unique identifier for the **legal case**. One case can contain multiple decisions (e.g. a procedural ruling followed by a merits ruling). |
| **Source / method** | Scraped from NALUS. The case identifier groups all decisions belonging to the same case file. |
| **Type / format** | `string` — e.g. `"Pl. ÚS 5/12"`, `"I. ÚS 234/20"`. Standard CCC citation format: `"formation. ÚS number/year"` |

---

## `date_decision`

| | |
|---|---|
| **Definition** | Date on which the CCC issued the decision. |
| **Source / method** | Scraped from NALUS metadata. |
| **Type / format** | `date`, format `YYYY-MM-DD` — e.g. `"2021-03-15"` |

---

## `date_submission`

| | |
|---|---|
| **Definition** | Date on which the case was submitted (filed) to the CCC. Together with `date_decision` it allows computation of proceeding length. |
| **Source / method** | Scraped from NALUS metadata. |
| **Type / format** | `date`, format `YYYY-MM-DD` — e.g. `"2020-06-01"` |

---

## `composition`

| | |
|---|---|
| **Definition** | List of all judges who sat on the panel deciding the case. |
| **Source / method** | **Regex-mined from the full text** of decisions. Paulík used regular expressions to extract judge names from the text corpus. ⚠️ Paulík notes this is **not entirely reliable for the first decade** (~1993–2003) due to irregularities in early decision formatting. |
| **Type / format** | `nested-list` of objects, each containing `judge_name` (`string`) and `judge_id` (`string`). E.g. `[{"judge_name": "Pavel Rychetský", "judge_id": "jr_01"}, ...]` |

---

## `formation`

| | |
|---|---|
| **Definition** | The judicial formation (body) that handled the case — either the full court (plénum) or one of the three-member chambers (senáty). |
| **Source / method** | Scraped from NALUS metadata. |
| **Type / format** | `string` — e.g. `"Plénum"`, `"I. senát"`, `"II. senát"`, `"III. senát"`, `"IV. senát"` |

---

## `judge_rapporteur_name`

| | |
|---|---|
| **Definition** | Full name of the judge rapporteur (soudce zpravodaj) — the judge primarily responsible for drafting the decision. |
| **Source / method** | Scraped from NALUS metadata. A corresponding `judge_rapporteur_id` exists in Paulík's dataset for joining with the judges background table. |
| **Type / format** | `string` — e.g. `"Kateřina Šimáčková"`, `"Vojtěch Šimíček"` |

---

## `type_verdict`

| | |
|---|---|
| **Definition** | The type(s) of verdict (výrok) the CCC reached. A single decision may contain multiple verdicts. Key values: `vyhověno` (granted), `zamítnuto` (rejected on merits), `odmítnuto` (rejected on admissibility), `procesní` (procedural). Each verdict also has a `verdict_ground` (merits / admissibility / procedural). |
| **Source / method** | Scraped from NALUS. The CCC database records the individual verdicts per decision. |
| **Type / format** | `nested-list` of objects with `verdict_type` (`string`) and `verdict_ground` (`string`). E.g. `[{"verdict_type": "vyhověno", "verdict_ground": "merits"}, {"verdict_type": "odmítnuto", "verdict_ground": "admissibility"}]` |

---

## `type_decision`

| | |
|---|---|
| **Definition** | The form of the decision document. Three possible values: **Nález** (judgment on the merits), **Usnesení** (resolution — typically admissibility or procedural), **Stanovisko pléna** (plenary opinion for unifying case law). |
| **Source / method** | Scraped from NALUS metadata. |
| **Type / format** | `string` — one of `"Nález"`, `"Usnesení"`, `"Stanovisko pléna"` |

---

## `type_proceedings`

| | |
|---|---|
| **Definition** | Type of constitutional proceedings as defined by the Constitution and the Act on the CCC. Determines the subject-matter jurisdiction. |
| **Source / method** | Scraped from NALUS metadata. |
| **Type / format** | `string` — e.g. `"O ústavních stížnostech"` (constitutional complaints), `"O zrušení zákonů a jiných právních předpisů"` (abstract review), `"O souladu mezinárodních smluv"` (international treaties review), `"Ve sporech o rozsah kompetencí státních orgánů a orgánů územní samosprávy"` (separation of powers conflicts), etc. |

---

## `grounds`

| | |
|---|---|
| **Definition** | Overall grounds of the decision — a summary categorisation derived from the individual verdicts. If any verdict is on merits → `"merits"`. If all verdicts are on admissibility → `"admissibility"`. If all verdicts are procedural → `"procedural"`. |
| **Source / method** | **Derived / inferred** by Paulík from the `type_verdict` data — not directly scraped. This is a computed variable applying a hierarchical rule (merits > admissibility > procedural) over all verdicts in a decision. |
| **Type / format** | `string` — one of `"merits"`, `"admissibility"`, `"procedural"` |

---

## `disputed_act`

| | |
|---|---|
| **Definition** | The legal act(s) being challenged / reviewed in the decision. Each decision can dispute multiple acts. Paulík also auto-classified each act into a `disputed_act_type`. |
| **Source / method** | The disputed act text is **scraped from NALUS**. The `disputed_act_type` classification (e.g. court decision, administrative decision, statute, municipal statute, government act) was **added automatically** by Paulík (likely rule-based classification). |
| **Type / format** | `nested-list` of objects with `disputed_act` (`string`) and `disputed_act_type` (`string`). E.g. `[{"disputed_act": "rozsudek Nejvyššího soudu ze dne ...", "disputed_act_type": "court_decision"}]`. Possible type values: `"court_decision"`, `"administrative_decision"`, `"statute"`, `"municipal_statute"`, and various government/context-specific acts. |

---

## `applicant`

| | |
|---|---|
| **Definition** | The applicant party (navrhovatel) who brought the case before the CCC. Subset of the broader `parties` dataset, filtered to `party_type == "applicant"`. |
| **Source / method** | Scraped from NALUS. The CCC database records all parties with their role. Paulík's dataset also includes `party_kind` — whether the applicant is a natural person, legal person, state authority, court, or state prosecution. |
| **Type / format** | `nested-list` of objects with `party` (`string`, full name/specification), `party_type` (`string`, always `"applicant"` here), `party_kind` (`string`). E.g. `[{"party": "Jan Novák", "party_type": "applicant", "party_kind": "natural_person"}]` |

---

## `concerned_body`

| | |
|---|---|
| **Definition** | The body whose decision or act is under CCC review (vedlejší účastník / dotčený orgán). Subset of the broader `parties` dataset, filtered to `party_type == "concerned_body"`. Note: CCC proceedings are not adversarial in form. |
| **Source / method** | Scraped from NALUS. Same source as `applicant` — the CCC database records both sides. |
| **Type / format** | `nested-list` of objects with `party` (`string`), `party_type` (`string`, always `"concerned_body"` here), `party_kind` (`string`). E.g. `[{"party": "Nejvyšší soud", "party_type": "concerned_body", "party_kind": "court"}]` |

---

## `separate_opinion`

| | |
|---|---|
| **Definition** | Information about judges who attached a separate (dissenting or concurring) opinion. Includes which judge dissented and whether they dissented alone or jointly with others (`dissenting_group`). |
| **Source / method** | **Hybrid.** Whether a judge attached a separate opinion comes from **NALUS metadata**. Additional details (grouping of joint dissents, etc.) were **regex-mined from the decision texts**. ⚠️ Like `composition`, reliability is lower for the first decade. |
| **Type / format** | `nested-list` of objects with `dissenting_judge_name` (`string`), `dissenting_judge_id` (`string`), and `dissenting_group` (`numeric` — judges who dissented together share the same group number). E.g. `[{"dissenting_judge_name": "Jan Musil", "dissenting_judge_id": "jm_01", "dissenting_group": 1}]` |

---

## `url_address`

| | |
|---|---|
| **Definition** | Direct URL to the decision detail page on the NALUS database. |
| **Source / method** | Scraped from NALUS — the URL of the decision's detail page. |
| **Type / format** | `string` (URL) — e.g. `"https://nalus.usoud.cz/Search/..."` |

---

## Source summary

| Source | Variables |
|---|---|
| **NALUS (scraped metadata)** | `doc_id`, `case_id`, `date_decision`, `date_submission`, `formation`, `judge_rapporteur_name`, `type_verdict`, `type_decision`, `type_proceedings`, `disputed_act` (text), `applicant`, `concerned_body`, `separate_opinion` (basic flag), `url_address` |
| **Regex from decision texts** | `composition`, `separate_opinion` (grouping detail) |
| **Derived / computed** | `grounds` (inferred from `type_verdict`), `disputed_act_type` (auto-classified) |

---

## Notes for our coder

1. **NALUS is the primary source.** Most variables come directly from the CCC's database at `nalus.usoud.cz`. Paulík's scraper is written in R (see `scripts/` in [his repo](https://github.com/stepanpaulik/ccc_dataset/tree/main/scripts)).
2. **Regex extraction from texts is fragile.** Paulík himself warns that `composition` and parts of `separate_opinion` are unreliable for ~1993–2003 due to inconsistent formatting in early decisions. We should verify or improve the regex patterns.
3. **`grounds` is not a raw variable** — it is computed from the verdicts using a hierarchical rule. We can either replicate this logic or store only the raw verdicts and compute on the fly.
4. **Nested-list variables** (`composition`, `type_verdict`, `disputed_act`, `applicant`, `concerned_body`, `separate_opinion`) represent one-to-many relationships at the decision level. Paulík stores these as nested JSON lists. We should decide on our target format (flat/relational tables vs. nested JSON) early.
