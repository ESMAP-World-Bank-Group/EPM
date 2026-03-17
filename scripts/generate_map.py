#!/usr/bin/env python3
"""
Generate EPM deployment bubble map from Excel/CSV data.

Usage:
    python scripts/generate_map.py

Input:  docs/data/epm_deployments.xlsx  (or .csv)
Output: docs/assets/epm_map.html

To update the map: edit the input file, re-run this script, rebuild the site.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
INPUT_XLSX = ROOT / "docs" / "data" / "epm_deployments.xlsx"
INPUT_CSV  = ROOT / "docs" / "data" / "epm_deployments.csv"
OUTPUT     = ROOT / "docs" / "assets" / "epm_map.html"

AMBER  = "#E8A800"  # since 2024  — matches site accent color
NAVY   = "#1b3a5c"  # before 2024 — matches site tab/primary color
LIGHT  = "#D5E6F3"  # non-EPM land — matches site background tint

# ── Country centroids {name: (lat, lon)} ──────────────────────────────────────
CENTROIDS = {
    # West Africa
    "Senegal": (14.50, -14.45), "The Gambia": (13.44, -15.31),
    "Gambia": (13.44, -15.31), "Guinea-Bissau": (11.80, -15.18),
    "Guinea": (10.84, -10.93), "Sierra Leone": (8.46, -11.78),
    "Liberia": (6.43, -9.43), "Ivory Coast": (7.54, -5.55),
    "Cote d'Ivoire": (7.54, -5.55), "Ghana": (7.94, -1.02),
    "Togo": (8.62, 0.82), "Benin": (9.31, 2.32),
    "Nigeria": (9.08, 8.68), "Niger": (17.61, 8.08),
    "Mali": (17.57, -3.99), "Burkina Faso": (12.36, -1.53),
    "Burkina-Faso": (12.36, -1.53), "Mauritania": (21.01, -10.94),
    "Cape Verde": (16.00, -24.01), "Cabo Verde": (16.00, -24.01),
    # Central Africa
    "Cameroon": (5.70, 12.35), "Chad": (15.45, 18.73),
    "Central African Republic": (6.61, 20.94),
    "Democratic Republic of the Congo": (-4.04, 21.76),
    "Democratic Republic of Congo": (-4.04, 21.76),
    "DR Congo": (-4.04, 21.76), "DRC": (-4.04, 21.76),
    "Republic of the Congo": (-0.23, 15.83),
    "Republic of Congo": (-0.23, 15.83), "Congo": (-0.23, 15.83),
    "Gabon": (-0.80, 11.61), "Equatorial Guinea": (1.65, 10.27),
    "Sao Tome and Principe": (0.41, 6.61),
    # East Africa
    "Ethiopia": (9.15, 40.49), "Kenya": (0.02, 37.91),
    "Tanzania": (-6.37, 34.89), "Uganda": (1.37, 32.29),
    "Rwanda": (-1.94, 29.87), "Burundi": (-3.37, 29.92),
    "South Sudan": (6.88, 31.30), "Sudan": (12.86, 30.22),
    "Djibouti": (11.83, 42.59), "Eritrea": (15.18, 39.78),
    "Somalia": (5.15, 46.20),
    # Southern Africa
    "Mozambique": (-18.67, 35.53), "Zimbabwe": (-19.02, 29.15),
    "Zambia": (-13.13, 27.85), "Malawi": (-13.25, 34.30),
    "Madagascar": (-20.28, 46.87), "South Africa": (-30.56, 22.94),
    "Namibia": (-22.96, 18.49), "Botswana": (-22.33, 24.68),
    "Lesotho": (-29.61, 28.23), "Eswatini": (-26.52, 31.47),
    "Swaziland": (-26.52, 31.47), "Angola": (-11.20, 17.87),
    "Comoros": (-11.88, 43.87), "Mauritius": (-20.35, 57.55),
    "Seychelles": (-4.68, 55.49),
    # North Africa
    "Morocco": (31.79, -7.09), "Algeria": (28.03, 1.66),
    "Tunisia": (34.00, 9.02), "Libya": (26.34, 17.23),
    "Egypt": (26.82, 30.80),
    # MENA
    "Saudi Arabia": (23.89, 45.08), "Yemen": (15.55, 48.52),
    "Oman": (21.51, 55.92), "United Arab Emirates": (23.42, 53.85),
    "UAE": (23.42, 53.85), "Qatar": (25.35, 51.18),
    "Kuwait": (29.31, 47.48), "Bahrain": (26.07, 50.56),
    "Iraq": (33.22, 43.68), "Jordan": (31.24, 36.51),
    "Lebanon": (33.85, 35.86), "Syria": (34.80, 38.99),
    "Iran": (32.43, 53.69), "Palestine": (31.95, 35.30),
    "West Bank and Gaza": (31.95, 35.30),
    # South Asia
    "India": (20.59, 78.96), "Pakistan": (30.38, 69.35),
    "Bangladesh": (23.68, 90.36), "Nepal": (28.39, 84.12),
    "Bhutan": (27.51, 90.43), "Sri Lanka": (7.87, 80.77),
    "Maldives": (3.20, 73.22), "Afghanistan": (33.94, 67.71),
    # Southeast Asia
    "Myanmar": (21.92, 95.96), "Thailand": (15.87, 100.99),
    "Vietnam": (14.06, 108.28), "Cambodia": (12.57, 104.99),
    "Laos": (19.86, 102.50), "Lao PDR": (19.86, 102.50),
    "Malaysia": (4.21, 108.96), "Indonesia": (-0.79, 113.92),
    "Philippines": (12.88, 121.77), "Timor-Leste": (-8.87, 125.73),
    "East Timor": (-8.87, 125.73),
    # Pacific
    "Papua New Guinea": (-6.31, 143.96), "Fiji": (-17.71, 178.07),
    "Solomon Islands": (-9.64, 160.16), "Vanuatu": (-15.38, 166.96),
    "Samoa": (-13.76, -172.10), "Tonga": (-21.18, -175.20),
    "Kiribati": (1.87, -157.36), "Marshall Islands": (7.10, 171.18),
    "Nauru": (-0.52, 166.93), "Tuvalu": (-8.52, 179.20),
    "Palau": (7.52, 134.58),
    # Central Asia
    "Kazakhstan": (48.02, 66.92), "Uzbekistan": (41.38, 64.59),
    "Kyrgyzstan": (41.20, 74.77), "Kyrgyz Republic": (41.20, 74.77),
    "Tajikistan": (38.86, 71.28), "Turkmenistan": (38.97, 59.56),
    "Mongolia": (46.86, 103.85),
    # Caucasus
    "Georgia": (42.32, 43.36), "Armenia": (40.07, 45.04),
    "Azerbaijan": (40.14, 47.58),
    # Europe / Balkans
    "Turkey": (38.96, 35.24), "Turkiye": (38.96, 35.24),
    "Ukraine": (48.38, 31.17), "Poland": (51.92, 19.15),
    "North Macedonia": (41.61, 21.74), "Macedonia": (41.61, 21.74),
    "Albania": (41.15, 20.17), "Kosovo": (42.56, 20.89),
    "Bosnia and Herzegovina": (44.17, 17.68),
    "Serbia": (44.02, 21.01), "Moldova": (47.41, 28.37),
    "Belarus": (53.71, 27.95),
    # Latin America
    "Mexico": (23.63, -102.55), "Guatemala": (15.78, -90.23),
    "Belize": (17.19, -88.50), "Honduras": (15.20, -86.24),
    "El Salvador": (13.79, -88.90), "Nicaragua": (12.87, -85.21),
    "Costa Rica": (9.75, -83.75), "Panama": (8.54, -80.78),
    "Colombia": (4.57, -74.30), "Venezuela": (6.42, -66.59),
    "Guyana": (4.86, -58.93), "Suriname": (3.92, -56.03),
    "Ecuador": (-1.83, -78.18), "Peru": (-9.19, -75.02),
    "Brazil": (-14.24, -51.93), "Bolivia": (-16.29, -63.59),
    "Paraguay": (-23.44, -58.44), "Chile": (-35.68, -71.54),
    "Argentina": (-38.42, -63.62), "Uruguay": (-32.52, -55.77),
    "Dominican Republic": (18.74, -70.16), "Haiti": (18.97, -72.29),
    "Jamaica": (18.11, -77.30), "Cuba": (21.52, -77.78),
    "Trinidad and Tobago": (10.69, -61.22), "Barbados": (13.19, -59.54),
    "Grenada": (12.11, -61.68), "Saint Lucia": (13.91, -60.97),
    "Dominica": (15.41, -61.37), "Antigua and Barbuda": (17.07, -61.80),
    # Alternate spellings found in data
    "Kirgizstan": (41.20, 74.77), "PNG": (-6.31, 143.96),
    "Lao": (19.86, 102.50), "Salvador": (13.79, -88.90),
    "Montenegro": (42.71, 19.37),
}


def load_data() -> pd.DataFrame:
    if INPUT_XLSX.exists():
        df = pd.read_excel(INPUT_XLSX)
    elif INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)
    else:
        sys.exit(
            f"No input file found.\n"
            f"Expected: {INPUT_XLSX}\n"
            f"      or: {INPUT_CSV}"
        )
    df.columns        = df.columns.str.strip()
    df["Country"]     = df["Country"].str.strip()
    df["Before 2024"] = pd.to_numeric(df["Before 2024"], errors="coerce").fillna(0).astype(int)
    df["After 2024"]  = pd.to_numeric(df["After 2024"],  errors="coerce").fillna(0).astype(int)
    df["Total"]       = df["Before 2024"] + df["After 2024"]
    df["Recent"]      = df["After 2024"] > 0
    return df


def make_map(df: pd.DataFrame) -> go.Figure:
    df["hover"] = df.apply(
        lambda r: (
            f"<b>{r['Country']}</b><br>"
            f"Total studies: <b>{r['Total']}</b><br>"
            f"Before 2024: {r['Before 2024']}<br>"
            f"Since 2024: {r['After 2024']}"
        ),
        axis=1,
    )

    former = df[~df["Recent"]]
    recent = df[df["Recent"]]

    # ── Before 2024 (navy) ────────────────────────────────────────────────────
    trace_former = go.Choropleth(
        locations=former["Country"],
        locationmode="country names",
        z=[1] * len(former),
        text=former["hover"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=[[0, NAVY], [1, NAVY]],
        showscale=False,
        marker=dict(line=dict(color="white", width=0.6)),
        name="Before 2024",
        showlegend=True,
    )

    # ── Since 2024 (amber) ────────────────────────────────────────────────────
    trace_recent = go.Choropleth(
        locations=recent["Country"],
        locationmode="country names",
        z=[1] * len(recent),
        text=recent["hover"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=[[0, AMBER], [1, AMBER]],
        showscale=False,
        marker=dict(line=dict(color="white", width=0.6)),
        name="Since 2024",
        showlegend=True,
    )

    n_countries = len(df)
    n_total     = int(df["Total"].sum())
    n_recent    = int(df["Recent"].sum())

    fig = go.Figure(data=[trace_former, trace_recent])
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            showland=True,
            landcolor=LIGHT,
            showocean=True,
            oceancolor="#F5FAFF",
            showlakes=False,
            showcountries=True,
            countrycolor="white",
            projection_type="natural earth",
            bgcolor="white",
            lataxis=dict(range=[-57, 78]),
            lonaxis=dict(range=[-175, 180]),
        ),
        legend=dict(
            orientation="h",
            x=0.02,
            xanchor="left",
            y=0.04,
            font=dict(size=12, color="#2d3d52", family="Inter, Arial, sans-serif"),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#c5d8ee",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=10),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#2d3d52"),
        title=dict(
            text=(
                f"{n_total} studies &nbsp;·&nbsp; "
                f"{n_countries} countries &nbsp;·&nbsp; "
                f"{n_recent} with recent deployments"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=13, color="#666666"),
        ),
    )
    return fig


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df  = load_data()
    fig = make_map(df)
    fig.write_html(
        OUTPUT,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": False, "responsive": True},
    )
    print(f"Map written to: {OUTPUT}")
    print(f"  {len(df)} countries, {df['Total'].sum()} total studies")


if __name__ == "__main__":
    main()
