"""Paso 1 — Demanda horaria del CENACE al formato del pipeline de días representativos.

OBJETO
    Transformar el registro horario del CENACE, donde cada GCR ocupa una
    columna, en una tabla larga `zone, month, day, hour, value` — el formato
    que espera `representativedays_pipeline.py`.

ENTRADA
    cenace_balance_2025.csv
    Fuente: CENACE, Demanda Pronosticada y Balance del SEN, Mercado del Día en
    Adelanto, agregado sobre el año 2025.
    Columnas: Year, Month, Day, Hour, y luego una columna de MW por GCR.
    8 760 líneas, es decir 365 días × 24 horas.

SALIDA
    input/load.csv — columnas `zone, month, day, hour, value`, 78 840 líneas.

CORRESPONDENCIA DE ZONAS
    El CENACE llama BCA y BCS a las dos Baja California; el modelo las llama
    SIBC y SIBCS. Las otras siete GCR llevan el mismo nombre de un lado y del
    otro. Ninguna agregación, ninguna desagregación: correspondencia uno a uno.

ELECCIÓN DEL AÑO
    2025 es el último año completo disponible. Sólo se usa después la FORMA
    horaria: los niveles vienen de pDemandForecast, calibrado sobre el PLADESE.
    La elección del año no afecta por tanto el volumen de demanda del modelo,
    solamente su reparto en el tiempo.

TRATAMIENTO DE LOS VALORES FALTANTES
    Los huecos se rellenan por interpolación lineal sobre la serie horaria de
    la zona en cuestión, y luego se señalan en el reporte de control. El
    archivo 2025 tiene uno solo.

CONTROLES IMPRESOS
    número de líneas, de días y de zonas; valores faltantes antes y después;
    energía anual y punta por GCR; punta coincidente del SEN y su relación con
    la suma de las puntas zonales.
"""

import sys
from pathlib import Path

import pandas as pd

ICI = Path(__file__).resolve().parent
ENTREE = Path(
    r"C:\Users\wb623661\OneDrive - WBG\Mexico\Data 2026"
    r"\CENACE - Demanda horaria 2025\cenace_balance_2025.csv"
)
SORTIE = ICI / "input" / "load.csv"

# CENACE -> nomenclatura del modelo
ZONES = {
    "BCA": "SIBC",
    "BCS": "SIBCS",
    "CEN": "CEN",
    "NES": "NES",
    "NOR": "NOR",
    "NTE": "NTE",
    "OCC": "OCC",
    "ORI": "ORI",
    "PEN": "PEN",
}

INDEX = ["Year", "Month", "Day", "Hour"]


def preparer(entree: Path = ENTREE, sortie: Path = SORTIE, verbeux: bool = True) -> pd.DataFrame:
    """Lee el registro del CENACE, lo pasa a formato largo, controla y guarda."""
    brut = pd.read_csv(entree)
    colonnes_zones = [c for c in brut.columns if c not in INDEX]

    inconnues = set(colonnes_zones) - set(ZONES)
    if inconnues:
        raise ValueError(f"Zonas no reconocidas en la entrada: {sorted(inconnues)}")
    manquantes = set(ZONES) - set(colonnes_zones)
    if manquantes:
        raise ValueError(f"Zonas esperadas ausentes de la entrada: {sorted(manquantes)}")

    if verbeux:
        print("PASO 1 — DEMANDA HORARIA")
        print(f"   entrada : {entree.name}")
        print(f"   {len(brut)} líneas, {len(colonnes_zones)} zonas, "
              f"{brut.groupby(['Month', 'Day']).ngroups} días")

    # --- valores faltantes, rellenados por zona
    trous = int(brut[colonnes_zones].isna().sum().sum())
    if trous:
        detail = brut[colonnes_zones].isna().sum()
        for z, n in detail[detail > 0].items():
            lignes = brut.index[brut[z].isna()]
            for i in lignes:
                print(f"   hueco rellenado : {z}, {brut.at[i, 'Month']:.0f}/"
                      f"{brut.at[i, 'Day']:.0f} h{brut.at[i, 'Hour']:.0f}")
        brut[colonnes_zones] = brut[colonnes_zones].interpolate(
            method="linear", limit_direction="both"
        )
    restants = int(brut[colonnes_zones].isna().sum().sum())
    if restants:
        raise ValueError(f"Quedan {restants} valores faltantes después de la interpolación.")
    if verbeux:
        print(f"   valores faltantes : {trous} rellenados, {restants} restantes")

    # --- paso al formato largo
    long = brut.melt(
        id_vars=INDEX, value_vars=colonnes_zones, var_name="zone", value_name="value"
    )
    long["zone"] = long["zone"].map(ZONES)
    long = long.rename(columns={"Month": "month", "Day": "day", "Hour": "hour"})
    # El CENACE numera las horas de 1 a 24; el pipeline de días
    # representativos espera de 0 a 23. Se convierte aquí, en el origen.
    long["hour"] = long["hour"].astype(int) - 1
    long = long[["zone", "month", "day", "hour", "value"]]
    long = long.sort_values(["zone", "month", "day", "hour"]).reset_index(drop=True)

    attendu = len(ZONES) * 8760
    if len(long) != attendu:
        raise ValueError(f"Se produjeron {len(long)} líneas, se esperaban {attendu}.")

    sortie.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(sortie, index=False)

    if verbeux:
        print(f"   salida : {sortie.relative_to(ICI)} — {len(long)} líneas")
        print()
        _controler(brut, colonnes_zones)

    return long


def _controler(brut: pd.DataFrame, colonnes_zones) -> None:
    """Imprime la tabla de control por GCR y los agregados del SEN."""
    print("CONTROL POR GCR")
    print(f"   {'zona':<7}{'energía GWh':>13}{'punta MW':>12}{'valle MW':>11}{'fact. carga':>14}")
    print("   " + "-" * 57)
    total = 0.0
    for c in colonnes_zones:
        energie = brut[c].sum() / 1000
        total += energie
        print(f"   {ZONES[c]:<7}{energie:>13,.0f}{brut[c].max():>12,.0f}"
              f"{brut[c].min():>11,.0f}{100 * brut[c].mean() / brut[c].max():>13.1f}%")
    print("   " + "-" * 57)
    print(f"   {'SEN':<7}{total:>13,.0f}")
    print()

    coincidente = brut[colonnes_zones].sum(axis=1).max()
    somme_zonale = sum(brut[c].max() for c in colonnes_zones)
    print("CONTROL DEL SEN")
    print(f"   punta coincidente          {coincidente:>10,.0f} MW")
    print(f"   suma de las puntas zonales {somme_zonale:>10,.0f} MW")
    print(f"   relación                   {somme_zonale / coincidente:>10.4f}")
    print()
    print("   Referencias PLADESE 2024 : consumo bruto 359 807 GWh,")
    print("   demanda máxima integrada del SEN 55 528 MW.")


if __name__ == "__main__":
    try:
        preparer()
    except Exception as exc:  # pragma: no cover
        print(f"FALLO : {exc}", file=sys.stderr)
        raise
