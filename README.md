# Ghostwriting at the CCC

## 1. Original Dataset and Codebook

- **Original dataset** [**(GitHub):**](https://github.com/stepanpaulik/ccc_dataset)

- [**Codebook**] [**(Zenodo):**](https://zenodo.org/records/10591110)

## 2. Inspiration and Related Literature

- [**Rosenthal & Yoon (2011), legal scholarship**](https://scholarship.law.cornell.edu/clr/vol96/iss6/11/)
  *Judicial Ghostwriting: Authorship on the Supreme Court*

- [**Rosenthal & Yoon (2011), statistical methodology**](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-5/issue-1/Detecting-multiple-authorship-of-United-States-Supreme-Court-legal-decisions/10.1214/10-AOAS378.full)  
  *Detecting Multiple Authorship of United States Supreme Court Legal Decisions*  

- [**Avraham et al. (2025)**](https://academic.oup.com/jla/article/17/1/2/8098035?login=false)
  
- https://ocnk.ff.cuni.cz/cs/ - zeptat se na dyachronní synchronní corpusy a na literaturu ohledně nich

## 3. Data Processing and Subset Construction

The main purpose of the preprocessing pipeline (see `Ghostwriters.Rmd`) is to prepare decision texts for stylometric and authorship-related analysis.

Two cleaned subsets are created from the original CCC dataset:

### `subset`

This dataset is designed for the assessment of court rulings.

to do:
- remove non-substantive text components
- isolate the reasoning sections of decisions, excluding headers, procedural parts, and ancillary materials

### `subset_disent2`

This dataset focuses exclusively on dissenting opinions and is used to construct individual-level textual **fingerprints**.

## 4. misc

- in [Paulík (2024)](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E3BA7816D8C221C3A13CF843E2526BA1/S2164657024000123a.pdf/div-class-title-the-czech-constitutional-court-database-div.pdf),
  footnote no. 13 _"I have been able to partition the text into an implicit structure such as the heading, procedure history, parties’ arguments, or court arguments using a supervised machine learning algorithm. However, the decisions do not lend themselves to a simple and reliable regex partitioning. The only clear general rule is that the first paragraph most of the time contains the heading of the decision, in which the composition as well as the parties of the case can be located."_
  Comments from the author: _"Ale nebylo to nic fancy, dneska by to určitě šlo líp nějakým LLM. Ale tehdy to bylo basic word2vec (a jeho různé odnože) reprezentace textu, vytvoření trénovacího datasetu a následné použití tehdy to byly tuším random forests, který vycházely nejlíp, nakonec jen nějaké zajištění kontinuity (to dost zvýšilo přesnost - že pokud je nějaký odstavec obklopen odstavci stejné kategorie, tak to přepíše tu predikovanou hodnotu)."_
- budeme sbírat předchozí rozhodnutí u soudců? co akademici? a hlavně, **advokáti??**
