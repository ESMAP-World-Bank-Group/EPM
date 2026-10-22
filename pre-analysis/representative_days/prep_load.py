"""Étape 1 — Demande horaire du CENACE au format du pipeline des jours représentatifs.

OBJET
    Transformer le relevé horaire du CENACE, où chaque GCR occupe une colonne,
    en un tableau long `zone, month, day, hour, value` — le format attendu par
    `representativedays_pipeline.py`.

ENTRÉE
    cenace_balance_2025.csv
    Source : CENACE, Demanda Pronosticada y Balance del SEN, Mercado del Día en
    Adelanto, agrégé sur l'année 2025.
    Colonnes : Year, Month, Day, Hour, puis une colonne de MW par GCR.
    8 760 lignes, soit 365 jours × 24 heures.

SORTIE
    input/load.csv — colonnes `zone, month, day, hour, value`, 78 840 lignes.

CORRESPONDANCE DES ZONES
    Le CENACE nomme les deux Basses-Californies BCA et BCS ; le modèle les
    appelle SIBC et SIBCS. Les sept autres GCR portent le même nom de part et
    d'autre. Aucune agrégation, aucune désagrégation : correspondance une à une.

CHOIX DE L'ANNÉE
    2025 est la dernière année complète disponible. Seule la FORME horaire est
    utilisée par la suite : les niveaux proviennent de pDemandForecast, calé sur
    le PLADESE. Le choix de l'année n'affecte donc pas le volume de demande du
    modèle, seulement sa répartition dans le temps.

TRAITEMENT DES VALEURS MANQUANTES
    Les trous sont comblés par interpolation linéaire sur la série horaire de la
    zone concernée, puis signalés dans le rapport de contrôle. Le fichier 2025
    en compte un seul.

CONTRÔLES IMPRIMÉS
    nombre de lignes, de jours et de zones ; valeurs manquantes avant et après ;
    énergie annuelle et pointe par GCR ; pointe coïncidente du SEN et rapport à
    la somme des pointes zonales.
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

# CENACE -> nomenclature du modèle
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
    """Lit le relevé CENACE, le met au format long, contrôle et enregistre."""
    brut = pd.read_csv(entree)
    colonnes_zones = [c for c in brut.columns if c not in INDEX]

    inconnues = set(colonnes_zones) - set(ZONES)
    if inconnues:
        raise ValueError(f"Zones non reconnues dans l'entrée : {sorted(inconnues)}")
    manquantes = set(ZONES) - set(colonnes_zones)
    if manquantes:
        raise ValueError(f"Zones attendues absentes de l'entrée : {sorted(manquantes)}")

    if verbeux:
        print("ÉTAPE 1 — DEMANDE HORAIRE")
        print(f"   entrée : {entree.name}")
        print(f"   {len(brut)} lignes, {len(colonnes_zones)} zones, "
              f"{brut.groupby(['Month', 'Day']).ngroups} jours")

    # --- valeurs manquantes, comblées par zone
    trous = int(brut[colonnes_zones].isna().sum().sum())
    if trous:
        detail = brut[colonnes_zones].isna().sum()
        for z, n in detail[detail > 0].items():
            lignes = brut.index[brut[z].isna()]
            for i in lignes:
                print(f"   trou comblé : {z}, {brut.at[i, 'Month']:.0f}/"
                      f"{brut.at[i, 'Day']:.0f} h{brut.at[i, 'Hour']:.0f}")
        brut[colonnes_zones] = brut[colonnes_zones].interpolate(
            method="linear", limit_direction="both"
        )
    restants = int(brut[colonnes_zones].isna().sum().sum())
    if restants:
        raise ValueError(f"{restants} valeurs manquantes subsistent après interpolation.")
    if verbeux:
        print(f"   valeurs manquantes : {trous} comblées, {restants} restantes")

    # --- passage au format long
    long = brut.melt(
        id_vars=INDEX, value_vars=colonnes_zones, var_name="zone", value_name="value"
    )
    long["zone"] = long["zone"].map(ZONES)
    long = long.rename(columns={"Month": "month", "Day": "day", "Hour": "hour"})
    # Le CENACE numérote les heures de 1 à 24 ; le pipeline des jours
    # représentatifs attend 0 à 23. On convertit ici, à la source.
    long["hour"] = long["hour"].astype(int) - 1
    long = long[["zone", "month", "day", "hour", "value"]]
    long = long.sort_values(["zone", "month", "day", "hour"]).reset_index(drop=True)

    attendu = len(ZONES) * 8760
    if len(long) != attendu:
        raise ValueError(f"{len(long)} lignes produites, {attendu} attendues.")

    sortie.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(sortie, index=False)

    if verbeux:
        print(f"   sortie : {sortie.relative_to(ICI)} — {len(long)} lignes")
        print()
        _controler(brut, colonnes_zones)

    return long


def _controler(brut: pd.DataFrame, colonnes_zones) -> None:
    """Imprime le tableau de contrôle par GCR et les agrégats du SEN."""
    print("CONTRÔLE PAR GCR")
    print(f"   {'zone':<7}{'énergie GWh':>13}{'pointe MW':>12}{'creux MW':>11}{'fact. charge':>14}")
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
    print("CONTRÔLE DU SEN")
    print(f"   pointe coïncidente         {coincidente:>10,.0f} MW")
    print(f"   somme des pointes zonales  {somme_zonale:>10,.0f} MW")
    print(f"   rapport                    {somme_zonale / coincidente:>10.4f}")
    print()
    print("   Repères PLADESE 2024 : consommation brute 359 807 GWh,")
    print("   demande maximale intégrée du SEN 55 528 MW.")


if __name__ == "__main__":
    try:
        preparer()
    except Exception as exc:  # pragma: no cover
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        raise
