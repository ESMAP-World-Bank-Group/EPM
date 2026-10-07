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
3. Validate the whole catalog: `python pre-analysis/catalog/validate.py`
4. Commit alongside the CSV and provenance update.

The filename must equal the `id` field — `source_id` citations in `provenance.yaml`
resolve by filename. Renaming a source means renaming its citations in the same commit.

## Validation

`validate.py` checks three things, and runs in CI (`.github/workflows/catalog.yml`):

- every `sources/*.yaml` validates against the schema, and its `id` matches its filename;
- every `source_id` cited in a `provenance.yaml` resolves to a catalog entry — a dangling
  citation is an **error**, because it means a number in the model has no traceable origin;
- catalog entries never cited by any provenance file — a **warning**, since a source may
  legitimately be documented before it is used.

Pass `--strict` to turn warnings into failures.

## Two axes, not one

`type` is the **access level** — who may receive the data (`open_source`, `internal_wb`,
`client_confidential`, `assumption`). It is what governs publication.

`category` is the **nature** of the source (`public_database`, `official_statistics`,
`study_document`, `derived`, `modelled`) and carries no access implication. A series
derived from public databases is `open_source` with `category: derived`.

These were conflated until 2026-08-14: entries used `internal`, `public_database`,
`official_statistics` and `derived` as `type` values, none of which the schema accepted.

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
python pre-analysis/catalog/generate_docs.py --deployment data_test
# → writes epm/input/data_test/DATA_SOURCES.md
```
