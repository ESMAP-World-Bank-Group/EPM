"""Paso 2 — Perfiles horarios solar y eólico por GCR.

OBJETO
    Producir, para cada una de las 9 regiones de control, un factor de carga
    horario del fotovoltaico y del eólico sobre el año 2025, en el formato que
    espera `representativedays_pipeline.py`.

CADENA EN TRES TIEMPOS
    1. localizar   — un punto por GCR, baricentro de las capacidades instaladas
    2. descargar   — Renewables.ninja, vía `pipelines/vre_pipeline.py` sin tocar
    3. corregir    — reajuste a la hora local de cada zona

1. LOCALIZACIÓN — baricentro ponderado por la capacidad instalada
    Fuente: Global Energy Monitor, *Global Integrated Power Tracker*,
    septiembre de 2025, pestaña `Power facilities`, filtrada sobre México, las
    centrales solares y eólicas en operación, geolocalizadas.
    El solar se filtra a 5 MW como mínimo: el PLADESE cuenta la generación
    distribuida aparte, y falsearía el baricentro.

    Cada central se asigna a su GCR por su Estado. La correspondencia viene
    del PLADESE **T 4.9**, *Requerimientos de capacidad de particulares*, que
    publica para cada proyecto su GCR y su Estado — 21 Estados así cubiertos.
    Los demás no pertenecen sin ambigüedad más que a una sola GCR, salvo tres
    casos resueltos y documentados en ETAT_GCR más abajo.

    El baricentro reemplaza la elección de un punto por el analista: es
    reproducible y verificable. La función `gap_rninja_coordinates` del
    pipeline, que retiene la central más grande de un PAÍS, no sirve aquí.

    Tres zonas no tienen ningún parque eólico registrado — CEN, NOR, NTE.
    Reciben el baricentro solar de su propia GCR, que marca el centro del
    desarrollo renovable regional. Estas tres zonas no llevan ninguna capacidad
    eólica existente en el modelo; el perfil sólo sirve ahí para los
    candidatos.

2. DESCARGA — Renewables.ninja, reanálisis MERRA-2
    Pfenninger S. e Staffell I. (2016), *Long-term patterns of European PV
    output using 30 years of validated hourly reanalysis and satellite data*,
    Energy 114, 1251-1265, y el artículo compañero sobre el eólico.
    Salidas bajo licencia CC-BY: la cita es una condición de uso.

    Año 2025, el mismo que la demanda horaria del CENACE preparada en el
    paso 1. La correlación entre demanda y renovables es uno de los criterios
    que la selección de los días representativos debe preservar: las dos series
    deben cubrir el mismo año.

    Fotovoltaico — seguidor de un eje, configuración dominante de las grandes
    centrales mexicanas; inclinación igual a la latitud con tope de 30°,
    acimut al sur, pérdidas de sistema 10 %.

    Eólico — Vestas V90 de 2 MW, buje a 100 m, es decir 314 W/m² de potencia
    específica. Es una CALIBRACIÓN, no una descripción del parque. La máquina
    por defecto del pipeline, una Gamesa G114 a 196 W/m², da un factor de carga
    nacional de 49 % contra 30,4 % observado: Renewables.ninja restituye una
    máquina nueva, perfectamente disponible, en un sitio perfectamente elegido,
    mientras que lo observado integra envejecimiento, indisponibilidades,
    estelas y recortes. Se compararon cinco curvas sobre los dos yacimientos
    principales; la V90 es la que devuelve el nacional al nivel observado.

3. CORRECCIÓN HORARIA
    Renewables.ninja devuelve las marcas de tiempo en UTC: su parámetro
    `local_time` no tiene efecto sobre el formato JSON, lo que se verificó
    comparando las dos respuestas, estrictamente idénticas. Sin corrección, el
    pico solar caería de seis a ocho horas demasiado tarde respecto a la
    demanda.

    Se aplica un desplazamiento circular del huso horario estándar de cada
    zona. México suprimió el horario de verano en 2022, los husos son por tanto
    fijos, con una excepción no modelada: Baja California todavía lo observa,
    por estar alineada con California — una hora durante ocho meses, zona SIBC
    únicamente.

CONTROLES
    factor de carga nacional ponderado, contra el PLADESE 2024, T 2.1 y T 2.2
        eólico   7 512 MW   19 987 GWh   ->  30,4 %
        solar    7 961 MW   18 640 GWh   ->  26,7 %
    hora del pico solar, que debe caer entre las 11 h y las 14 h en cada zona.

SALIDA
    input/pv.csv e input/wind.csv — `zone, month, day, hour, value`,
    78 840 líneas cada uno, valores en [0, 1].
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

# Huso horario estándar respecto a UTC.
FUSEAU = {"CEN": -6, "ORI": -6, "OCC": -6, "NTE": -6, "NES": -6, "PEN": -6,
          "NOR": -7, "SIBCS": -7, "SIBC": -8}
ZONES = list(FUSEAU)

# Estado -> GCR. Los 21 primeros vienen del PLADESE T 4.9; los siguientes no
# tienen ambigüedad; los tres últimos están resueltos y justificados.
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
    "hidalgo": "CEN",    # T 4.9 lo cita en Central y en Occidental; mayoría en Central
    "tlaxcala": "ORI",   # enclavado en Puebla, misma región de transmisión
    "guerrero": "ORI",   # ligado a Oriental por Ixtapa - Lázaro Cárdenas
}

# Capacidad instalada por GCR, pGenDataInput Status 1 (MW), para la ponderación.
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
    """Calcula, por GCR y por tecnología, el baricentro de las capacidades GEM."""
    if not GEM.exists():
        raise FileNotFoundError(f"Archivo GEM no encontrado : {GEM}")
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
    # zonas sin parque eólico : se retoma el baricentro solar de la misma GCR
    reprises = []
    for z in ZONES:
        if ("wind", z) not in pts and ("solar", z) in pts:
            lat, lon, _, _ = pts[("solar", z)]
            pts[("wind", z)] = (lat, lon, 0.0, 0)
            reprises.append(z)

    if verbeux:
        print("1. LOCALIZACIÓN — baricentros ponderados por la capacidad (GEM 09/2025)")
        print(f"   {'GCR':<7}{'lat PV':>9}{'lon PV':>10}{'MW':>8}"
              f"{'lat viento':>11}{'lon viento':>10}{'MW':>8}")
        print("   " + "-" * 64)
        for z in ZONES:
            p, e = pts.get(("solar", z)), pts.get(("wind", z))
            print(f"   {z:<7}{p[0]:>9.2f}{p[1]:>10.2f}{p[2]:>8.0f}"
                  f"{e[0]:>11.2f}{e[1]:>10.2f}{e[2]:>8.0f}")
        print("   " + "-" * 64)
        if reprises:
            print(f"   eólico tomado del baricentro solar, ningún parque registrado : "
                  f"{', '.join(reprises)}")
        print()
    return pts


def _long(df: pd.DataFrame) -> pd.DataFrame:
    """Formato largo, llevado a la hora local por desplazamiento circular."""
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
    # El pipeline de días representativos espera horas de 0 a 23,
    # convención que Renewables.ninja ya respeta.
    out["hour"] = out["hour"].astype(int)
    return out.sort_values(["zone", "month", "day", "hour"]).reset_index(drop=True)


def preparer(verbeux: bool = True, seulement: str = None) -> dict:
    jeton = load_api_token("renewables_ninja")
    if not jeton:
        raise RuntimeError(
            "Token de Renewables.ninja no encontrado. Se espera en config/api_tokens.ini, "
            "sección [api_tokens], clave renewables_ninja."
        )
    if verbeux:
        print("PASO 2 — PERFILES SOLAR Y EÓLICO")
        print(f"   Renewables.ninja, reanálisis MERRA-2, año {ANNEE}, 9 GCR")
        print()
    pts = barycentres(verbeux)

    if verbeux:
        print("2. DESCARGA")
    if seulement == "wind":
        pv = pd.read_csv(ICI / "input" / "pv.csv")
        if verbeux:
            print(f"   solar    tomado de input/pv.csv, {len(pv)} líneas")
        blocs = None
    else:
        blocs = []
    for z in (ZONES if blocs is not None else []):
        lat, lon, _, _ = pts[("solar", z)]
        tilt = min(abs(lat), TILT_MAX)
        if verbeux:
            print(f"   solar    {z:<6}{lat:7.2f}{lon:9.2f}   inclinación {tilt:.0f}°")
        df = get_renewables_ninja(
            {z: (lat, lon)}, "solar", ANNEE, ANNEE + 1, api_token=jeton,
            export_csv=False, tracking=1, tilt=tilt, azim=180,
            system_loss=0.1, capacity=1,
        )
        if df.empty:
            raise RuntimeError(f"Respuesta solar vacía para {z} : ¿límite de la API?")
        blocs.append(_long(df))
        time.sleep(3)
    if blocs is not None:
        pv = pd.concat(blocs, ignore_index=True)
    (ICI / "input").mkdir(parents=True, exist_ok=True)
    pv.to_csv(ICI / "input" / "pv.csv", index=False)
    if verbeux:
        print(f"   salida : input/pv.csv — {len(pv)} líneas, {pv.zone.nunique()} zonas")

    if verbeux:
        print(f"   eólico   9 zonas, {TURBINE.replace('+', ' ')}, buje {HAUTEUR} m")
    df = get_renewables_ninja(
        {z: (pts[("wind", z)][0], pts[("wind", z)][1]) for z in ZONES}, "wind",
        ANNEE, ANNEE + 1, api_token=jeton, export_csv=False,
        turbine=TURBINE, height=HAUTEUR, capacity=1,
    )
    wind = _long(df) if not df.empty else pd.DataFrame(columns=["zone"])

    # Reintento zona por zona de lo que falta : el límite de la API es una
    # ventana deslizante, las solicitudes se liberan con los minutos.
    for essai in range(1, 9):
        manquantes = [z for z in ZONES if z not in set(wind.zone)]
        if not manquantes:
            break
        if verbeux:
            print(f"   reintento {essai} — zonas faltantes : {', '.join(manquantes)}")
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
                    print(f"      {z} recuperada")
            time.sleep(8)

    manquantes = [z for z in ZONES if z not in set(wind.zone)]
    if manquantes:
        raise RuntimeError(
            "Zonas todavía ausentes después de los reintentos : " + ", ".join(manquantes)
        )
    wind = wind.sort_values(["zone", "month", "day", "hour"]).reset_index(drop=True)

    wind.to_csv(ICI / "input" / "wind.csv", index=False)
    if verbeux:
        print(f"   salida : input/wind.csv — {len(wind)} líneas, {wind.zone.nunique()} zonas")
    sorties = {"pv": pv, "wind": wind}

    if verbeux:
        _controler(sorties)
    return sorties


def _controler(sorties: dict) -> None:
    print()
    print("3. CONTROLES")
    print(f"   {'GCR':<7}{'FC PV':>8}{'FC viento':>10}{'pico PV':>9}{'1a h':>8}{'últ. h':>8}")
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
    print(f"   husos horarios : {'todos correctos' if ok else 'ANOMALÍA'}")
    print()
    print("   nacional ponderado por la capacidad instalada, contra PLADESE 2024")
    for tech, nom in (("pv", "solar"), ("wind", "eólico")):
        s, c = sorties[tech], CAPACITE[tech]
        fc = sum(s.loc[s.zone == z, "value"].mean() * c.get(z, 0) for z in ZONES)
        fc /= sum(c.values())
        print(f"      {nom:<9}{100 * fc:>7.1f}%   observado {100 * CIBLE[tech]:>5.1f}%"
              f"   desviación {100 * (fc - CIBLE[tech]) / CIBLE[tech]:>+6.1f} %")


if __name__ == "__main__":
    try:
        preparer(seulement="wind" if "--wind" in sys.argv else None)
    except Exception as exc:  # pragma: no cover
        print(f"FALLO : {exc}", file=sys.stderr)
        raise
