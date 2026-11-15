"""Paso 5 — Corrección y control de las salidas destinadas a EPM.

OBJETO
    Corregir el único defecto conocido del pipeline, y luego verificar que los
    tres archivos entregados a EPM son coherentes entre sí y con los datos de
    origen.

LA CORRECCIÓN — normalización de pDemandProfile por zona
    `utils.py`, función `_ensure_normalized_series`, divide la serie de demanda
    por su máximo GLOBAL, todas las zonas juntas. EPM espera un perfil cuyo
    máximo valga 1 para CADA zona, puesto que luego lo multiplica por la punta
    zonal de `pDemandForecast`.

    Sin corrección, y según las puntas medidas de 2025, OCC alcanzaría 1,00
    pero SIBCS se quedaría en 0,058: el modelo creería que Baja California Sur
    nunca demanda más de la vigésima parte de su potencia.
    La relación entre la zona más grande y la más pequeña es de 17.

    La corrección se hace AQUÍ y no antes: la misma serie normalizada sirve
    también para identificar el día de punta del sistema, y hay que conservar
    entonces los tamaños relativos de las regiones. Corregir en la entrada
    resolvería un problema y crearía otro.

LOS CONTROLES
    1. suma de pHours igual a 8 760 horas
    2. máximo de pDemandProfile igual a 1 para cada una de las 9 zonas
    3. zonas y días coherentes entre los tres archivos
    4. energía anual reconstituida, comparada con la medición del CENACE
    5. factores de carga solar y eólico reconstituidos, comparados con las
       series horarias de origen

    El control 4 es el más elocuente: reconstituye la energía anual de cada GCR
    a partir únicamente de los días tipo y de sus pesos, y luego la confronta
    con las 8 760 horas reales. Una desviación importante indicaría que los
    días retenidos no representan el año.

SALIDA
    output/epm/pDemandProfile.csv corregido en el sitio, conservándose el
    original bajo pDemandProfile_brut.csv.
"""

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ICI = Path(__file__).absolute().parent
EPM = ICI / "output" / "epm"
ENTREE = ICI / "input"
ZONES = ["CEN", "ORI", "OCC", "NOR", "NTE", "NES", "PEN", "SIBC", "SIBCS"]


def _heures(ph: pd.DataFrame) -> pd.Series:
    """Peso en horas de cada par estación / tipo de día."""
    t = [c for c in ph.columns if c.startswith("t")]
    cle = [c for c in ph.columns if c not in t][:2]
    return ph.set_index(cle)[t].sum(axis=1)


def corriger(verbeux: bool = True) -> pd.DataFrame:
    """Renormaliza pDemandProfile zona por zona."""
    f = EPM / "pDemandProfile.csv"
    brut = EPM / "pDemandProfile_brut.csv"
    shutil.copy2(f, brut)   # ídem : se recopia en cada ejecución, corrección idempotente
    d = pd.read_csv(brut)
    t = [c for c in d.columns if c.startswith("t")]
    zc = d.columns[0]
    for c in t:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if verbeux:
        print("CORRECCIÓN — normalización de pDemandProfile por zona")
        print(f"   {'zona':<8}{'máx antes':>12}{'máx después':>12}")
        print("   " + "-" * 32)
    for z, g in d.groupby(zc):
        m = g[t].max().max()
        if m and m > 0:
            d.loc[d[zc] == z, t] = (g[t] / m).values
        if verbeux:
            print(f"   {z:<8}{m:>12.4f}{1.0:>12.4f}")
    d.to_csv(f, index=False)
    if verbeux:
        print(f"   original conservado bajo {brut.name}")
        print()
    return d


JOURS_SAISON = {"Q1": 90, "Q2": 91, "Q3": 92, "Q4": 92}   # 2025, año no bisiesto


def corriger_pHours(verbeux: bool = True) -> pd.DataFrame:
    """Devuelve los pesos de cada estación al número real de días del calendario.

    DEFECTO CORREGIDO
        El pipeline asigna a cada día extremo el peso de su grupo de
        clasificación, pero no retira ese mismo número de días del lote
        sometido a la optimización. Las dos cuentas no coinciden entonces, y la
        suma de pHours se aparta de 8 760 horas.
        Con un juego de datos, los grupos pueden caer justo por casualidad y el
        defecto queda invisible; deja de estarlo en cuanto los datos cambian.
        Constatado aquí: Q4 pesaba 107 días en lugar de 92.

    CORRECCIÓN
        Renormalización POR ESTACIÓN, proporcional. Los pesos relativos entre
        días de una misma estación se conservan, y cada estación recupera su
        número real de días. Es la corrección más neutra: no privilegia ni los
        días extremos ni los días optimizados.
    """
    f = EPM / "pHours.csv"
    brut = EPM / "pHours_brut.csv"
    # Se recopia en CADA ejecución : una copia de seguridad guardada de una
    # ejecución anterior haría corregir pesos caducos tras una nueva selección.
    # La operación es idempotente -- renormalizar un archivo ya correcto da un
    # factor de 1 -- así que relanzar sólo el control no tiene efecto.
    shutil.copy2(f, brut)
    ph = pd.read_csv(brut)
    t = [c for c in ph.columns if c.startswith("t")]
    cs, cd = [c for c in ph.columns if c not in t][:2]
    if verbeux:
        print("CORRECCIÓN - pesos de los días por estación")
        print(f"   {'estación':<8}{'antes':>9}{'real':>8}{'factor':>10}")
        print("   " + "-" * 35)
    for q, g in ph.groupby(cs):
        av = g[t].sum().sum() / 24
        cible = JOURS_SAISON.get(str(q), av)
        if av:
            ph.loc[ph[cs] == q, t] = (g[t] * cible / av).values
        if verbeux:
            print(f"   {str(q):<8}{av:>9.0f}{cible:>8.0f}{cible / av if av else 1:>10.4f}")
    ph.to_csv(f, index=False)
    if verbeux:
        print(f"   total {ph[t].sum().sum():.0f} h   original conservado bajo {brut.name}")
        print()
    return ph


def controler(verbeux: bool = True) -> bool:
    ok = True
    ph = pd.read_csv(EPM / "pHours.csv")
    dp = pd.read_csv(EPM / "pDemandProfile.csv")
    vp = pd.read_csv(EPM / "pVREProfile.csv")
    w = _heures(ph)

    print("CONTROLES")
    # 1 — suma de las horas
    total = w.sum()
    bon = abs(total - 8760) < 0.5
    ok &= bon
    print(f"   1. suma de pHours             {total:>10.0f} h   esperado 8 760   "
          f"{'ok' if bon else 'ANOMALÍA'}")
    print(f"      {len(ph)} días tipo, es decir {total / 24:.0f} días")

    # 2 — máximo por zona
    t = [c for c in dp.columns if HEURE.match(c)]
    zc = dp.columns[0]
    for c in t:
        dp[c] = pd.to_numeric(dp[c], errors="coerce")
    mx = dp.groupby(zc)[t].max().max(axis=1)
    bon = bool(((mx - 1).abs() < 1e-6).all())
    ok &= bon
    print(f"   2. máx de pDemandProfile      {mx.min():.4f} a {mx.max():.4f}      "
          f"{'ok, 1,00 en todas partes' if bon else 'ANOMALÍA'}")

    # 3 — coherencia de las zonas
    zd, zv = set(dp[zc]), set(vp[vp.columns[0]])
    bon = zd == set(ZONES) and zv == set(ZONES)
    ok &= bon
    print(f"   3. zonas                      demanda {len(zd)}, VRE {len(zv)}      "
          f"{'ok' if bon else 'ANOMALÍA : ' + str(set(ZONES) ^ zd)}")

    # 4 — energía anual reconstituida
    print()
    print("   4. energía anual reconstituida, contra la medición del CENACE")
    ld = pd.read_csv(ENTREE / "load.csv")
    reel = ld.groupby("zone").value.sum() / 1000  # GWh
    pointe = ld.groupby("zone").value.max()
    cle = [c for c in dp.columns if c not in t][:3]
    print(f"      {'zona':<8}{'reconstituido':>13}{'medido':>11}{'desv.':>9}")
    print("      " + "-" * 41)
    ecarts = []
    for z in ZONES:
        g = dp[dp[zc] == z]
        e = sum(g[g[cle[1]] == q][g[cle[2]] == d][t].values.sum() * w.get((q, d), 0) / 24
                for q, d in w.index if len(g[(g[cle[1]] == q) & (g[cle[2]] == d)]))
        e = e * pointe[z] / 1000
        ec = 100 * (e - reel[z]) / reel[z]
        ecarts.append(abs(ec))
        print(f"      {z:<8}{e:>13,.0f}{reel[z]:>11,.0f}{ec:>8.1f}%")
    print("      " + "-" * 41)
    pire = max(ecarts)
    bon = pire < 10
    ok &= bon
    print(f"      desviación máxima {pire:.1f} %   {'ok' if bon else 'A EXAMINAR'}")

    # 5 — factores de carga VRE reconstituidos
    print()
    print("   5. factor de carga VRE reconstituido, contra la serie horaria")
    tv = [c for c in vp.columns if HEURE.match(c)]
    for c in tv:
        vp[c] = pd.to_numeric(vp[c], errors="coerce")
    cv = [c for c in vp.columns if c not in tv][:4]
    for tech, fichier in (("PV", "pv.csv"), ("Wind", "wind.csv")):
        src = pd.read_csv(ENTREE / fichier)
        s = vp[vp[cv[1]].astype(str).str.lower() == tech.lower()]
        if not len(s):
            continue
        num = sum(r[tv].mean() * w.get((r[cv[2]], r[cv[3]]), 0) for _, r in s.iterrows())
        den = len(s[cv[0]].unique()) * 8760
        print(f"      {tech:<6}{100 * num / den:>7.1f}%   serie horaria "
              f"{100 * src.value.mean():>6.1f}%")

    print()
    print(f"   -> {'TODOS LOS CONTROLES PASAN' if ok else 'ANOMALÍAS, ver arriba'}")
    return ok


# --------------------------------------------------------------------------- #
# Exportación al formato que espera EPM
# --------------------------------------------------------------------------- #

MODELE = ICI.parent.parent / "epm" / "input" / "data_mexico_2026"
EXPORT = ICI / "output" / "epm_modele"

# El pipeline y el modelo no nombran las columnas de la misma manera.
RENOMMAGE = {"zone": "z", "season": "q", "daytype": "d", "fuel": "tech"}
# Las columnas horarias son t1 a t24; un simple startswith("t") atraparía
# también la columna "tech" y la duplicaría en la salida.
HEURE = re.compile(r"^t\d+$")
# Las tecnologías tampoco.
TECHNOS = {"Wind": "OnshoreWind", "PV": "PV"}


def exporter(verbeux: bool = True) -> dict:
    """Pone las tres salidas al formato del modelo y las escribe en output/epm_modele.

    TRES TRANSPOSICIONES

    1. Nombres de columnas. El pipeline escribe `zone, season, daytype, fuel`;
       el modelo espera `z, q, d, tech`.

    2. Nombre de tecnología. El pipeline escribe `Wind`, el modelo
       `OnshoreWind`, que es el valor de la columna `tech` en pGenDataInput.

    3. Perfil de la central solar de concentración. main.gms línea 662 pone
           pCSPProfile(cs,q,d,t) = sum(..., pVREProfile(z,tech,q,d,t))
       El perfil CSP se lee por tanto en pVREProfile bajo la tecnología
       `CSPPlant`. El pipeline sólo produce PV y eólico: se añade una línea
       CSPPlant para SIBCS, única zona con una central CSP, retomando el
       perfil fotovoltaico de esa zona.

       Es una APROXIMACIÓN declarada. El recurso de una central de
       concentración es la irradiancia directa normal, mientras que el
       fotovoltaico responde a la irradiancia global inclinada. Las dos están
       fuertemente correlacionadas con cielo despejado, y el seguidor de un eje
       elegido para el PV acerca todavía más las dos. La diferencia se da sobre
       todo en los días nublados, cuando la CSP produce menos de lo que este
       perfil deja creer.
    """
    EXPORT.mkdir(parents=True, exist_ok=True)
    sorties = {}

    ph = pd.read_csv(EPM / "pHours.csv").rename(columns=RENOMMAGE)
    sorties["pHours.csv"] = ph

    dp = pd.read_csv(EPM / "pDemandProfile.csv").rename(columns=RENOMMAGE)
    sorties["pDemandProfile.csv"] = dp[["z", "q", "d"] + [c for c in dp.columns if HEURE.match(c)]]

    vp = pd.read_csv(EPM / "pVREProfile.csv").rename(columns=RENOMMAGE)
    vp["tech"] = vp["tech"].map(lambda x: TECHNOS.get(x, x))
    csp = vp[(vp.tech == "PV") & (vp.z == "SIBCS")].copy()
    csp["tech"] = "CSPPlant"
    vp = pd.concat([vp, csp], ignore_index=True)
    sorties["pVREProfile.csv"] = vp[["z", "tech", "q", "d"] + [c for c in vp.columns if HEURE.match(c)]]

    if verbeux:
        print()
        print("EXPORTACIÓN AL FORMATO DEL MODELO")
    for nom, d in sorties.items():
        d.to_csv(EXPORT / nom, index=False, encoding="utf-8-sig")
        if verbeux:
            print(f"   {nom:<22}{len(d):>5} líneas   columnas {', '.join(d.columns[:4])}")
    if verbeux:
        print(f"   escrito en {EXPORT.relative_to(ICI)}")
        print()
        print("   a copiar hacia el modelo :")
        print(f"      pHours.csv          -> {MODELE.name}/")
        print(f"      pDemandProfile.csv  -> {MODELE.name}/load/")
        print(f"      pVREProfile.csv     -> {MODELE.name}/supply/")
    return sorties


def main():
    if not (EPM / "pHours.csv").exists():
        raise FileNotFoundError(f"Salidas de EPM no encontradas en {EPM}")
    print("PASO 5 — CORRECCIÓN Y CONTROL")
    print()
    corriger_pHours()
    corriger()
    ok = controler()
    exporter()
    resume = ICI / "output" / "representative_days_summary.csv"
    if resume.exists():
        print()
        print("DÍAS RETENIDOS")
        print(pd.read_csv(resume).to_string(index=False))
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as exc:  # pragma: no cover
        print(f"FALLO : {exc}", file=sys.stderr)
        raise
