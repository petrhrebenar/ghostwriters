#libraries

library(ggplot2)
library(dplyr)
library(tidyr)
library(writexl)
library(readxl)
library(stringr)
library(xml2)
library(purrr)




#load datasets
texts <- read.csv("/Users/petrhrebenar/Desktop/Rko/DIPLOMKA/ccc_database/csv/ccc_texts.csv")

meta <- read.csv("/Users/petrhrebenar/Desktop/Rko/DIPLOMKA/ccc_database/csv/ccc_metadata.csv")
  meta$date_decision <- as.Date(meta$date_decision, format="%Y-%m-%d")
  meta$date_submission <- as.Date(meta$date_submission, format="%Y-%m-%d")
  meta$date_publication <- as.Date(meta$date_publication, format="%Y-%m-%d")

separate_opinions <- read.csv("/Users/petrhrebenar/Desktop/Rko/DIPLOMKA/ccc_database/csv/ccc_separate_opinions.csv")

verdicts <- read.csv("/Users/petrhrebenar/Desktop/Rko/DIPLOMKA/ccc_database/csv/ccc_verdicts.csv")

##################################
### subsetting - text analysis ###
##################################

subset <- texts %>%
  select(doc_id, text) %>%
  left_join(
    meta %>%
      select(
        doc_id,
        date_decision,
        date_submission,
        type_decision,
        importance,
        judge_rapporteur_name,
        judge_rapporteur_id,
        type_verdict,
        grounds,
        separate_opinion,
        formation,
        outcome,
        length_proceeding
      ),
    by = "doc_id"
  )

##################################
### filtering data for disents ###
##################################

subset_disent <- texts %>%
  select(doc_id, text) %>%
  left_join(
    meta %>%
      select(
        doc_id,
        date_decision,
        date_submission,
        type_decision,
        separate_opinion,
        formation,
        length_proceeding
      ),
    by = "doc_id") %>%  
  mutate(separate_opinion = as.character(separate_opinion),
  separate_opinion = str_trim(separate_opinion)) %>%
  # drop NAs
  filter(!is.na(separate_opinion), separate_opinion != "NA") %>%
  # drop rows where multiple judges disented
  filter(!str_detect(separate_opinion, "^c\\s*\\(.*\\)$"))

# filtering only disent text

# Extract everything AFTER the paragraph that contains the marker
extract_after_paragraph <- function(html, markers = c("Odlišné stanovisko")) {
  if (is.na(html)) return(NA_character_)
  # normalize non-breaking spaces
  html2 <- str_replace_all(html, fixed("&nbsp;"), " ")
  # convert to plain text (best-effort)
  plain <- tryCatch(
    xml_text(read_html(paste0("<body>", html2, "</body>"))),
    error = function(e) html2
  )
  # ensure markers is a character vector
  markers <- as.character(markers)
  # escape regex metacharacters in each marker
  escape_regex <- function(x) {
    str_replace_all(x, "([\\^$.|()\\[\\]{}*+?\\\\])", "\\\\\\1")
  }
  escaped <- vapply(markers, escape_regex, character(1))
  # build case-insensitive regex matching any of the markers
  # (?i) makes it case-insensitive
  pattern <- paste0("(?i)(", paste(escaped, collapse = "|"), ")")
  # locate first occurrence of any marker
  loc <- str_locate(plain, regex(pattern))
  if (is.na(loc[1])) return(NA_character_)
  # substring starting right after the marker occurrence
  after_marker <- str_sub(plain, loc[2] + 1)
  after_marker <- str_trim(after_marker)
  # find end of the paragraph that contains the marker:
  # look for the first blank line (one or more newline, maybe spaces) after the marker
  blank_loc <- str_locate(after_marker, "\\n\\s*\\n")
  if (!is.na(blank_loc[1])) {
    # everything AFTER that blank line
    result <- str_sub(after_marker, blank_loc[2] + 1)
  } else {
    # no blank line found — fallback: return the rest after the marker paragraph
    result <- after_marker
  }
  result <- str_trim(result)
  if (result == "") NA_character_ else result
}

# Apply to your already filtered dataset
subset_disent2 <- subset_disent %>%
  # create a new column (so you keep original fields), or overwrite if you prefer
  mutate(separate_opinion_extracted = map_chr(text, ~ extract_after_paragraph(.x, marker = "Odlišné stanovisko"))) %>%
  # optional: drop NA extracts (if you want to keep only rows where extraction succeeded)
  filter(!is.na(separate_opinion_extracted)) %>%
  # optional: remove R-list-like values "c(...)" if they appear in extracted text
  filter(!str_detect(separate_opinion_extracted, "^\\s*c\\s*\\(.*\\)\\s*$"))


#test to lost docs
lost_docs <- setdiff(
  subset_disent$doc_id,
  subset_disent2$doc_id
)

length(lost_docs) 
lost_docs

##################################################
# plot showing number of single-authored disents #
##################################################

subset_disent2 %>%
  count(separate_opinion) %>%
  ggplot(aes(x = reorder(separate_opinion, -n), y = n)) +
  geom_col() +
  geom_text(aes(label = n), vjust = -0.3, size = 3) +
  labs(
    x = "Dissenting judge (the only author)",
    y = "Number of observations"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
