"""
Generate the EPM coverage map (docs/assets/epm_map.html).

Re-run this script to refresh the map after updating DATA below.
    python docs/assets/make_epm_map.py

Each country has a count of studies "before 2025" and "since 2025".
"Recent" = at least one study since 2025 (blue, shaded by count); "Former" = only before (grey).
"""
import pycountry
import plotly.graph_objects as go

OUT = __file__.replace("make_epm_map.py", "epm_map.html")

# (country, studies before 2025, studies since 2025) — duplicates (same values) are de-duped.
DATA = [
    ("Senegal",3,2),("The Gambia",2,1),("Guinea-Bissau",1,1),("Guinea",1,3),("Ivory Coast",2,1),
    ("Sierra Leone",3,1),("Togo",2,1),("Benin",1,1),("Ghana",3,1),("Nigeria",2,1),("Niger",1,1),
    ("Mali",2,2),("Burkina-Faso",1,1),("Liberia",3,3),("Uzbekistan",2,0),("Kazakhstan",1,0),
    ("Kirgizstan",1,0),("Tajikistan",1,0),("Turkmenistan",1,0),("Afghanistan",2,1),("Pakistan",1,1),
    ("Morocco",2,0),("Algeria",1,0),("Tunisia",1,0),("Libya",1,1),("Egypt",2,2),("Sudan",3,1),
    ("Palestine",1,0),("Jordan",1,0),("Lebanon",2,1),("Syria",1,0),("Iraq",2,0),("Saudi Arabia",1,0),
    ("Bahrain",1,0),("Kuwait",1,0),("Qatar",1,0),("UAE",1,0),("Oman",1,0),("Yemen",1,0),
    ("Angola",0,2),("Botswana",1,2),("DRC",0,3),("Namibia",1,1),("South Africa",0,2),("Lesotho",0,1),
    ("Eswatini",1,3),("Zambia",0,1),("Mozambique",1,1),("Zimbabwe",1,2),("Malawi",2,1),("Tanzania",0,3),
    ("Burundi",0,3),("Rwanda",1,3),("Uganda",0,2),("Kenya",1,1),("Somalia",0,1),("Ethiopia",0,1),
    ("Eritrea",0,1),("South Sudan",0,1),("Djibouti",1,1),("India",1,2),("Bangladesh",2,2),("Nepal",0,2),
    ("Bhutan",1,1),("Sri Lanka",0,1),("Chad",1,1),("Turkiye",0,1),("PNG",1,0),("Myanmar",1,0),
    ("Lao",1,0),("Vietnam",2,0),("Ukraine",1,1),("Poland",1,0),("Kosovo",2,0),("Madagascar",1,0),
    ("Argentina",1,0),("Belize",1,0),("Bolivia",1,0),("Brazil",1,0),("Chile",1,0),("Colombia",2,0),
    ("Costa Rica",2,0),("Ecuador",1,0),("Salvador",1,0),("Guatemala",2,0),("Guyana",1,0),("Honduras",1,0),
    ("Mexico",1,1),("Nicaragua",1,0),("Panama",1,1),("Paraguay",1,0),("Peru",1,0),("Suriname",1,0),
    ("Uruguay",1,0),("Venezuela",1,0),("North Macedonia",1,0),("Bosnia",2,1),("Montenegro",1,0),
    ("Serbia",1,1),("Albania",1,0),("Indonesia",2,0),("Georgia",2,1),("Mauritania",2,1),("Cameroon",0,1),
    ("Gabon",1,2),("Equatorial Guinea",0,1),("Central African Republic",0,1),("Republic of the Congo",0,1),
    ("Armenia",0,1),("Azerbaijan",0,1),("Romania",0,1),("Bulgaria",0,1),
]

OVERRIDES = {
    "The Gambia":"GMB","Ivory Coast":"CIV","Burkina-Faso":"BFA","Kirgizstan":"KGZ","DRC":"COD",
    "Republic of the Congo":"COG","Eswatini":"SWZ","Turkiye":"TUR","PNG":"PNG","Lao":"LAO",
    "Salvador":"SLV","UAE":"ARE","North Macedonia":"MKD","Bosnia":"BIH","Syria":"SYR","Vietnam":"VNM",
    "Palestine":"PSE","Venezuela":"VEN","Bolivia":"BOL","Tanzania":"TZA","South Sudan":"SSD",
    "Guinea-Bissau":"GNB","Central African Republic":"CAF","Equatorial Guinea":"GNQ","Sri Lanka":"LKA",
    "Kirgizstan":"KGZ",
}

def iso3(name):
    if name in OVERRIDES:
        return OVERRIDES[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        try:
            return pycountry.countries.search_fuzzy(name)[0].alpha_3
        except LookupError:
            return None

# de-dupe (keep one row per country) and resolve ISO3
rows = {}
for name, before, after in DATA:
    code = iso3(name)
    if code is None:
        print("  [skip] no ISO3 for:", name)
        continue
    rows[code] = (name, before, after)

former = {c: v for c, v in rows.items() if v[2] == 0}
recent = {c: v for c, v in rows.items() if v[2] > 0}

def hover(name, before, after):
    return f"<b>{name}</b><br>Before 2025: {before}<br>Since 2025: {after}"

BLUE   = "#1E6DB8"   # Recent (at least one study since 2025)
YELLOW = "#E9B73A"   # Former (studies only before 2025)

fig = go.Figure()
# Former (yellow)
fig.add_trace(go.Choropleth(
    locations=list(former), z=[0]*len(former),
    colorscale=[[0, YELLOW], [1, YELLOW]], showscale=False,
    text=[hover(*v) for v in former.values()], hoverinfo="text",
    marker_line_color="white", marker_line_width=0.4,
))
# Recent (blue)
fig.add_trace(go.Choropleth(
    locations=list(recent), z=[0]*len(recent),
    colorscale=[[0, BLUE], [1, BLUE]], showscale=False,
    text=[hover(*v) for v in recent.values()], hoverinfo="text",
    marker_line_color="white", marker_line_width=0.4,
))
# Legend (two dummy markers — choropleth traces don't show in the legend)
fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode="markers",
    marker=dict(size=11, color=BLUE), name="Recent (since 2025)"))
fig.add_trace(go.Scattergeo(lon=[None], lat=[None], mode="markers",
    marker=dict(size=11, color=YELLOW), name="Earlier only (before 2025)"))

fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)", landcolor="#f4f6f9", showcountries=True,
                countrycolor="white", showland=True)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=480, paper_bgcolor="white",
                  dragmode=False,
                  legend=dict(orientation="h", x=0.5, xanchor="center", y=0.04,
                              bgcolor="rgba(255,255,255,0.75)", borderwidth=0,
                              font=dict(size=11)))

fig.write_html(OUT, include_plotlyjs="cdn", full_html=True,
               config={"displayModeBar": False, "scrollZoom": False})
print(f"OK -> {OUT}  ({len(rows)} countries: {len(recent)} recent, {len(former)} former)")
