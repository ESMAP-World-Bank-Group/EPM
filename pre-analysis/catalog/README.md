# EPM Data Source Catalog

One YAML file per reusable data source. Every file that adds or updates an EPM CSV
must add or update the matching catalog entry in the same commit.

## Structure

```
catalog/
  schema/
    source.schema.json    # JSON Schema — what a valid entry looks like
  sources/
    <id>.yaml             # one file per source
  generate_docs.py        # generates DATA_SOURCES.md for a deployment
  README.md               # this file
```

## Adding a new source

1. Copy an existing entry from `sources/` as a template.
2. Fill all required fields (see `schema/source.schema.json`).
3. Validate: `python -c "import jsonschema, yaml, json; jsonschema.validate(yaml.safe_load(open('sources/YOUR.yaml')), json.load(open('schema/source.schema.json')))"`
4. Commit alongside the CSV and provenance update.

## Method codes (used in provenance.yaml)

| Code | Meaning |
|------|---------|
| `DIRECT` | Value taken directly from source without transformation |
| `INTERP` | Linear interpolation between two known anchor points |
| `EXTRAP` | Extrapolation beyond last known point at constant rate |
| `PROXY_XX` | Proxy from country XX (e.g. `PROXY_GEO` = Georgia profile used) |
| `ASSUMED` | Engineering assumption — no direct data available |
| `CONVERTED` | Unit conversion only (e.g. EUR/GJ → USD/GJ) |

## Confidence levels

| Level | Meaning |
|-------|---------|
| `high` | Validated official data from a TSO, statistical office, or ENTSO-E |
| `medium` | Modeller elaboration (CESI, ESMAP) or interpolated from validated anchors |
| `low` | Proxy from another country or engineering default |

## Generating human-readable docs

```bash
python pre-analysis/catalog/generate_docs.py --deployment data_blacksea
# → writes epm/input/data_blacksea/DATA_SOURCES.md
```
