# Annotations (factgenie)

We use [factgenie](https://github.com/ufal/factgenie) for manual (human) and
LLM-based span annotations. **factgenie itself is not vendored into this repo** —
it is installed as an external tool. This directory holds only the artifacts we
own: input datasets, campaign definitions, and collected annotations.

## Layout

```
annotations/
  data/
    inputs/                # input datasets fed into factgenie (the texts to annotate)
    outputs/               # model/system outputs to be annotated
    datasets_TEMPLATE.yml  # dataset registration template (copy to datasets.yml)
  campaigns/               # campaign configs + collected annotations (human + LLM)
```

These directories are **symlinked into the external factgenie clone**
(`factgenie/data` -> `annotations/data`, `factgenie/campaigns` ->
`annotations/campaigns`), so anything created in the factgenie UI lands here and
is version-controlled.

## One-time setup

1. Clone factgenie *outside* this repo and install it in an isolated venv:

   ```bash
   git clone https://github.com/ufal/factgenie.git ~/ufal/factgenie
   python -m venv ~/ufal/factgenie/.venv
   source ~/ufal/factgenie/.venv/bin/activate
   pip install -e "~/ufal/factgenie[dev,deploy]"
   ```

2. Point factgenie's working dirs at this repo (run once, from repo root). The
   shipped `data/` content is copied into `annotations/data/` first so the
   templates are preserved:

   ```bash
   FG=~/ufal/factgenie/factgenie
   rm -rf "$FG/data" "$FG/campaigns"
   ln -s "$(pwd)/annotations/data"      "$FG/data"
   ln -s "$(pwd)/annotations/campaigns" "$FG/campaigns"
   ```

3. Create the config from the template and add API keys for LLM annotation
   (the config holds secrets and is **not** committed):

   ```bash
   cp ~/ufal/factgenie/factgenie/config/config_TEMPLATE.yml \
      ~/ufal/factgenie/factgenie/config/config.yml
   # then edit config.yml: set api_keys (OPENAI_API_KEY / ANTHROPIC_API_KEY / OLLAMA_API_KEY ...)
   ```

## Running

```bash
source ~/ufal/factgenie/.venv/bin/activate
factgenie run --host=127.0.0.1 --port 8890
# open http://127.0.0.1:8890
```

## Notes

- Do **not** commit `config.yml` or any API keys.
- Inputs typically come from our pipeline (e.g. `experiment_02`) exported into
  factgenie's expected dataset format under `annotations/data/`.
- See the factgenie wiki: Data Management, LLM Annotations, Crowdsourcing Annotations.
