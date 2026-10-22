"""Étape 2 — Profils horaires solaire et éolien par GCR.

OBJET
    Produire, pour chacune des 9 régions de contrôle, un facteur de charge
    horaire du photovoltaïque et de l'éolien sur l'année 2025, au format attendu
    par `representativedays_pipeline.py`.

CHAÎNE EN TROIS TEMPS
    1. localiser  — un point par GCR, barycentre des capacités installées
    2. télécharger — Renewables.ninja, via `pipelines/vre_pipeline.py` inchangé
    3. corriger    — remise à l'heure locale de chaque zone

1. LOCALISATION — barycentre pondéré par la capacité installée
    Source : Global Energy Monitor, *Global Integrated Power Tracker*,
    septembre 2025, onglet `Power facilities`, filtré sur le Mexique, les
    centrales solaires et éoliennes en opération, géolocalisées.
    Le solaire est filtré à 5 MW minimum : le PLADESE compte la génération
    distribuée à part, et elle fausserait le barycentre.

    Chaque centrale est rattachée à sa GCR par son État. La correspondance vient
    du PLADESE **T 4.9**, *Requerimientos de capacidad de particulares*, qui
    publie pour chaque projet sa GCR et son État — 21 États ainsi couverts.
    Les autres n'appartiennent sans ambiguïté qu'à une seule GCR, sauf trois
    cas tranchés et documentés dans ETAT_GCR ci-dessous.

    Le barycentre remplace le choix d'un point par l'analyste : il est
    reproductible et vérifiable. La fonction `gap_rninja_coordinates` du
    pipeline, qui retient la plus grosse centrale d'un PAYS, ne convient pas ici.

    Trois zones n'ont aucun parc éolien recensé — CEN, NOR, NTE. Elles
    reçoivent le barycentre solaire de leur propre GCR, qui marque le centre du
    développement renouvelable régional. Ces trois zones ne portent aucune
    capacité éolienne existante dans le modèle ; le profil n'y sert qu'aux
    candidats.

2. TÉLÉCHARGEMENT — Renewables.ninja, réanalyse MERRA-2
    Pfenninger S. et Staffell I. (2016), *Long-term patterns of European PV
    output using 30 years of validated hourly reanalysis and satellite data*,
    Energy 114, 1251-1265, et l'article compagnon sur l'éolien.
    Sorties sous licence CC-BY : la citation est une condition d'usage.

    Année 2025, la même que la demande horaire du CENACE préparée à l'étape 1.
    La corrélation entre demande et renouvelables est l'un des critères que la
    sélection des jours représentatifs doit préserver : les deux séries doivent
    porter sur la même année.

    Photovoltaïque — suiveur à un axe, configuration dominante des grandes
    centrales mexicaines ; inclinaison égale à la latitude plafonnée à 30°,
    azimut plein sud, pertes système 10 %.

    Éolien — Gamesa G80 de 2 MW, moyeu à 100 m, soit 398 W/m² de puissance
    spécifique. C'est un CALIBRAGE, pas une description du parc. La machine par
    défaut du pipeline, une Gamesa G114 à 196 W/m², donne un facteur de charge
    national de 49 % contre 30,4 % observé : Renewables.ninja restitue une
    machine neuve, parfaitement disponible, sur un site parfaitement choisi,
    quand l'observé intègre vieillissement, indisponibilités, sillages et
    écrêtements. Cinq courbes ont été comparées sur les deux gisements
    principaux ; la G80 est celle qui ramène le national au niveau observé.

3. CORRECTION HORAIRE
    Renewables.ninja renvoie les horodatages en UTC : son paramètre `local_time`
    est sans effet sur le format JSON, ce qui a été vérifié en comparant les deux
    réponses, strictement identiques. Sans correction, le pic solaire tomberait
    six à huit heures trop tard par rapport à la demande.

    On applique un décalage circulaire du décalage horaire standard de chaque
    zone. Le Mexique a supprimé l'heure d'été en 2022, les décalages sont donc
    fixes, à une exception non modélisée : la Basse-Californie l'observe encore,
    étant alignée sur la Californie — une heure sur huit mois, zone SIBC seule.

CONTRÔLES
    facteur de charge national pondéré, contre le PLADESE 2024, T 2.1 et T 2.2
        éolien   7 512 MW   19 987 GWh   ->  30,4 %
        solaire  7 961 MW   18 640 GWh   ->  26,7 %
    heure du pic solaire, qui doit tomber entre 11 h et 14 h dans chaque zone.

SORTIE
    input/pv.csv et input/wind.csv — `zone, month, day, hour, value`,
    78 840 lignes chacun, valeurs dans [0, 1].
"""

import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "pipelines"))

from vre_pipeline import get_renewables_ninja, load_api_token  # noqa: E402

ANNEE = 2025
GEM = ICI.parent / "dataset" / "Global-Integrated-Power-September-2025.xlsx"
SEUIL_SOLAIRE_MW = 5.0

# Décalage horaire standard par rapport à UTC.
FUSEAU = {"CEN": -6, "ORI": -6, "OCC": -6, "NTE": -6, "NES": -6, "PEN": -6,
          "NOR": -7, "SIBCS": -7, "SIBC": -8}
ZONES = list(FUSEAU)

# État -> GCR. Les 21 premiers viennent du PLADESE T 4.9 ; les suivants sont
# sans ambiguïté ; les trois derniers sont tranchés et justifiés.
ETAT_GCR = {
    "estado de mexico": "CEN", "mexico": "CEN", "state of mexico": "CEN",
    "coahuila": "NES", "coahuila de zaragoza": "NES", "tamaulipas": "NES",
    "chihuahua": "NTE",
    "guanajuato": "OCC", "guanajuto": "OCC", "jalisco": "OCC", "nayarit": "OCC",
    "queretaro": "OCC", "san luis potosi": "OCC", "zacatecas": "OCC",
    "chiapas": "ORI", "morelos": "ORI", "oaxaca": "ORI", "puebla": "ORI",
    "tabasco": "ORI", "veracruz": "ORI",
    "campeche": "PEN", "quintana roo": "PEN", "yucatan": "PEN",
    "sonora": "NOR", "sinaloa": "NOR", "durango": "NTE", "nuevo leon": "NES",
    "colima": "OCC", "michoacan": "OCC", "michoacan de ocampo": "OCC",
    "aguascalientes": "OCC", "baja california": "SIBC",
    "baja california sur": "SIBCS", "lower california south": "SIBCS",
    "ciudad de mexico": "CEN", "distrito federal": "CEN",
    "hidalgo": "CEN",    # T 4.9 le cite en Central et en Occidental ; majorité en Central
    "tlaxcala": "ORI",   # enclavé dans Puebla, même région de transmission
    "guerrero": "ORI",   # rattaché à Oriental par Ixtapa - Lázaro Cárdenas
}

# Capacité installée par GCR, pGenDataInput Status 1 (MW), pour la pondération.
CAPACITE = {
    "pv":   {"CEN": 120, "ORI": 828, "OCC": 2357, "NOR": 1784, "NTE": 1134,
             "NES": 1431, "PEN": 352, "SIBC": 41, "SIBCS": 91},
    "wind": {"CEN": 0, "ORI": 2480, "OCC": 640, "NOR": 2, "NTE": 0,
             "NES": 3523, "PEN": 245, "SIBC": 253, "SIBCS": 50},
}
CIBLE = {"pv": 18640 / (7961 * 8.760), "wind": 19987 / (7512 * 8.760)}

TILT_MAX = 30.0
TURBINE = (sys.argv[sys.argv.index("--turbine") + 1]
           if "--turbine" in sys.argv else "Vestas+V90+2000")
HAUTEUR = 100


def _sans_accent(s):
    s = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def barycentres(verbeux: bool = True) -> dict:
    """Calcule, par GCR et par technologie, le barycentre des capacités GEM."""
    if not GEM.exists():
        raise FileNotFoundError(f"Fichier GEM introuvable : {GEM}")
    d = pd.read_excel(GEM, sheet_name="Power facilities")
    col = [c for c in d.columns if "Subnational" in c][0]
    v = d[(d["Country/area"] == "Mexico") & d["Type"].isin(["solar", "wind"])
          & (d["Status"] == "operating") & d["Latitude"].notna()].copy()
    v = v[(v["Type"] == "wind") | (v["Capacity (MW)"] >= SEUIL_SOLAIRE_MW)]
    v["gcr"] = v[col].map(lambda e: ETAT_GCR.get(_sans_accent(e), "?"))

    pts = {}
    for tech in ("solar", "wind"):
        s = v[v.Type == tech]
        for z in ZONES:
            k = s[s.gcr == z]
            if len(k):
                w = k["Capacity (MW)"]
                pts[(tech, z)] = ((k.Latitude * w).sum() / w.sum(),
                                  (k.Longitude * w).sum() / w.sum(),
                                  w.sum(), len(k))
    # zones sans parc eolien : on reprend le barycentre solaire de la meme GCR
    reprises = []
    for z in ZONES:
        if ("wind", z) not in pts and ("solar", z) in pts:
            lat, lon, _, _ = pts[("solar", z)]
            pts[("wind", z)] = (lat, lon, 0.0, 0)
            reprises.append(z)

    if verbeux:
        print("1. LOCALISATION — barycentres pondérés par la capacité (GEM 09/2025)")
        print(f"   {'GCR':<7}{'lat PV':>9}{'lon PV':>10}{'MW':>8}"
              f"{'lat vent':>11}{'lon vent':>10}{'MW':>8}")
        print("   " + "-" * 64)
        for z in ZONES:
            p, e = pts.get(("solar", z)), pts.get(("wind", z))
            print(f"   {z:<7}{p[0]:>9.2f}{p[1]:>10.2f}{p[2]:>8.0f}"
                  f"{e[0]:>11.2f}{e[1]:>10.2f}{e[2]:>8.0f}")
        print("   " + "-" * 64)
        if reprises:
            print(f"   éolien repris du barycentre solaire, aucun parc recensé : "
                  f"{', '.join(reprises)}")
        print()
    return pts


def _long(df: pd.DataFrame) -> pd.DataFrame:
    """Format long, ramené à l'heure locale par décalage circulaire."""
    d = df.reset_index()
    val = [c for c in d.columns if c not in ("zone", "month", "day", "hour")]
    d = d.rename(columns={val[0]: "value"})[["zone", "month", "day", "hour", "value"]]
    d["value"] = pd.to_numeric(d["value"], errors="coerce").clip(0.0, 1.0)
    blocs = []
    for zone, g in d.groupby("zone", sort=False):
        g = g.sort_values(["month", "day", "hour"]).reset_index(drop=True)
        g = g[~((g.month == 2) & (g.day == 29))]
        g["value"] = np.roll(g["value"].to_numpy(), FUSEAU[zone])
        blocs.append(g)
    out = pd.concat(blocs, ignore_index=True)
    # Le pipeline des jours représentatifs attend des heures de 0 à 23,
    # convention que Renewables.ninja respecte déjà.
    out["hour"] = out["hour"].astype(int)
    return out.sort_values(["zone", "month", "day", "hour"]).reset_index(drop=True)


def preparer(verbeux: bool = True, seulement: str = None) -> dict:
    jeton = load_api_token("renewables_ninja")
    if not jeton:
        raise RuntimeError(
            "Jeton Renewables.ninja introuvable. Attendu dans config/api_tokens.ini, "
            "section [api_tokens], clé renewables_ninja."
        )
    if verbeux:
        print("ÉTAPE 2 — PROFILS SOLAIRE ET ÉOLIEN")
        print(f"   Renewables.ninja, réanalyse MERRA-2, année {ANNEE}, 9 GCR")
        print()
    pts = barycentres(verbeux)

    if verbeux:
        print("2. TÉLÉCHARGEMENT")
    if seulement == "wind":
        pv = pd.read_csv(ICI / "input" / "pv.csv")
        if verbeux:
            print(f"   solaire  repris de input/pv.csv, {len(pv)} lignes")
        blocs = None
    else:
        blocs = []
    for z in (ZONES if blocs is not None else []):
        lat, lon, _, _ = pts[("solar", z)]
        tilt = min(abs(lat), TILT_MAX)
        if verbeux:
            print(f"   solaire  {z:<6}{lat:7.2f}{lon:9.2f}   inclinaison {tilt:.0f}°")
        df = get_renewables_ninja(
            {z: (lat, lon)}, "solar", ANNEE, ANNEE + 1, api_token=jeton,
            export_csv=False, tracking=1, tilt=tilt, azim=180,
            system_loss=0.1, capacity=1,
        )
        if df.empty:
            raise RuntimeError(f"Réponse solaire vide pour {z} : limite d'API ?")
        blocs.append(_long(df))
        time.sleep(3)
    if blocs is not None:
        pv = pd.concat(blocs, ignore_index=True)
    (ICI / "input").mkdir(parents=True, exist_ok=True)
    pv.to_csv(ICI / "input" / "pv.csv", index=False)
    if verbeux:
        print(f"   sortie : input/pv.csv — {len(pv)} lignes, {pv.zone.nunique()} zones")

    if verbeux:
        print(f"   éolien   9 zones, {TURBINE.replace('+', ' ')}, moyeu {HAUTEUR} m")
    df = get_renewables_ninja(
        {z: (pts[("wind", z)][0], pts[("wind", z)][1]) for z in ZONES}, "wind",
        ANNEE, ANNEE + 1, api_token=jeton, export_csv=False,
        turbine=TURBINE, height=HAUTEUR, capacity=1,
    )
    wind = _long(df) if not df.empty else pd.DataFrame(columns=["zone"])

    # Reprise zone par zone de ce qui manque : la limite de l'API est une
    # fenetre glissante, les requetes se liberent au fil des minutes.
    for essai in range(1, 9):
        manquantes = [z for z in ZONES if z not in set(wind.zone)]
        if not manquantes:
            break
        if verbeux:
            print(f"   reprise {essai} — zones manquantes : {', '.join(manquantes)}")
        time.sleep(70)
        for z in manquantes:
            lat, lon = pts[("wind", z)][0], pts[("wind", z)][1]
            d1 = get_renewables_ninja(
                {z: (lat, lon)}, "wind", ANNEE, ANNEE + 1, api_token=jeton,
                export_csv=False, turbine=TURBINE, height=HAUTEUR, capacity=1,
            )
            if not d1.empty:
                wind = pd.concat([wind, _long(d1)], ignore_index=True)
                if verbeux:
                    print(f"      {z} recupere")
            time.sleep(8)

    manquantes = [z for z in ZONES if z not in set(wind.zone)]
    if manquantes:
        raise RuntimeError(
            "Zones toujours absentes apres reprises : " + ", ".join(manquantes)
        )
    wind = wind.sort_values(["zone", "month", "day", "hour"]).reset_index(drop=True)

    wind.to_csv(ICI / "input" / "wind.csv", index=False)
    if verbeux:
        print(f"   sortie : input/wind.csv — {len(wind)} lignes, {wind.zone.nunique()} zones")
    sorties = {"pv": pv, "wind": wind}

    if verbeux:
        _controler(sorties)
    return sorties


def _controler(sorties: dict) -> None:
    print()
    print("3. CONTRÔLES")
    print(f"   {'GCR':<7}{'FC PV':>8}{'FC vent':>10}{'pic PV':>9}{'1ere h':>8}{'der. h':>8}")
    print("   " + "-" * 50)
    ok = True
    for z in ZONES:
        p = sorties["pv"][sorties["pv"].zone == z]
        e = sorties["wind"][sorties["wind"].zone == z]
        g = p.groupby("hour").value.mean()
        nz = g[g > 0.001]
        bon = 10 <= g.idxmax() <= 13 and nz.index.min() >= 4 and nz.index.max() <= 20
        ok &= bon
        print(f"   {z:<7}{100 * p.value.mean():>7.1f}%{100 * e.value.mean():>9.1f}%"
              f"{g.idxmax():>9.0f}{nz.index.min():>8.0f}{nz.index.max():>8.0f}"
              f"{'' if bon else '   <<<'}")
    print("   " + "-" * 50)
    print(f"   fuseaux : {'tous corrects' if ok else 'ANOMALIE'}")
    print()
    print("   national pondéré par la capacité installée, contre PLADESE 2024")
    for tech, nom in (("pv", "solaire"), ("wind", "éolien")):
        s, c = sorties[tech], CAPACITE[tech]
        fc = sum(s.loc[s.zone == z, "value"].mean() * c.get(z, 0) for z in ZONES)
        fc /= sum(c.values())
        print(f"      {nom:<9}{100 * fc:>7.1f}%   observé {100 * CIBLE[tech]:>5.1f}%"
              f"   écart {100 * (fc - CIBLE[tech]) / CIBLE[tech]:>+6.1f} %")


if __name__ == "__main__":
    try:
        preparer(seulement="wind" if "--wind" in sys.argv else None)
    except Exception as exc:  # pragma: no cover
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        raise
