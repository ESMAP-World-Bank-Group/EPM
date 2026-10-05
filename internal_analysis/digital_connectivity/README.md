# Digital connectivity screening, Black Sea and South Caucasus

Internal screening of digital infrastructure in the Black Sea and South Caucasus,
and of its overlap with the energy and transport corridors already covered by the
Black Sea regional power trade study.

## How to reproduce

```bash
python fetch_data.py                  # downloads the open datasets into ./data (about 15 MB)
python build_note.py                  # writes digital_connectivity_screening.html here
python build_note.py --out "<folder>"  # and copies it to a shared folder
```

The note is shared through the team folder, so the working command is
`python build_note.py --out "<the Internal EPM analysis folder>"`. The path is not
hardcoded because it is specific to one machine.

`fetch_data.py` needs network access. `build_note.py` does not: it reads only
`./data` and the country boundaries in `epm/input/data_blacksea/extras/`. The note
is a single self-contained HTML file with every map and chart as inline SVG, so it
works offline and can be mailed as is.

`./data` is gitignored because every file in it is re-downloadable.

## Sources

| Dataset | Endpoint | Licence |
|---|---|---|
| Terrestrial fibre links | `bbmaps.itu.int/geoserver/ows`, layer `itu-geocatalogue:trx_geocatalogue` | ITU, attribution |
| Submarine cables and landing points | `submarinecablemap.com/api/v3` | CC BY-NC-SA 3.0 |
| Internet exchanges and facilities | `peeringdb.com/api` | Open, attribution |
| ICT indicators | `api.worldbank.org/v2` | CC BY 4.0 |
| High-voltage lines, pipelines and railways | Overpass API, OpenStreetMap | ODbL |

Two of these are non-commercial licences. Clear the licensing before reusing the
maps in any external deliverable.

## Caveats

The note states them in full. The short version: the ITU catalogue is mapped
backbone rather than total national fibre, the OpenStreetMap extracts cover 220 kV
and above and are truncated at the screening window, and the co-location metric in
section 7.2 is a rate between two independently mapped datasets rather than a
survey of installed optical ground wire.

Every route on the maps is real published geometry except one. The BSSC fibre has
no published alignment, so it is traced along the Caucasus Cable System, the
surveyed Poti to Balchik crossing in the same latitude band, with the landfalls
moved to Anaklia and Constanta. The captions say so wherever it is drawn. The
TRIPP corridor is the real closed railway alignment along the Aras from
OpenStreetMap, not a line between endpoints.

Overpass is a shared public service. If it times out, `fetch_data.py` tries three
mirrors in turn and the note falls back to the EPM zone topology for the master
map, which is a schematic abstraction and not real line routing.
