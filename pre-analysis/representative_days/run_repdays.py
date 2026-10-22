"""Étapes 3 et 4 — Sélection des jours représentatifs.

OBJET
    Réduire une année complète de données horaires — 8 760 heures, 9 régions,
    trois séries — à un petit nombre de journées types pondérées, puis les
    mettre au format d'EPM.

ENTRÉES
    input/load.csv   demande horaire par GCR, étape 1, source CENACE
    input/pv.csv     facteur de charge solaire, étape 2, source Renewables.ninja
    input/wind.csv   facteur de charge éolien, idem

SAISONS
    Quatre saisons calées sur les trimestres civils. Le découpage trimestriel a
    un sens physique au Mexique, où la pointe est franchement estivale, et il
    coïncide avec la saisonnalité que publie le PLADESE.
    On n'utilise donc pas `representativeseasons_pipeline.py`, qui déduirait les
    saisons par regroupement statistique : un découpage lisible et reproductible
    vaut mieux ici qu'un découpage optimal mais opaque.

NOMBRE DE JOURNÉES
    Trois journées optimisées par saison, auxquelles le pipeline ajoute
    automatiquement les journées extrêmes — ensoleillement minimal, vent
    minimal, demande maximale. Soit de l'ordre de vingt-quatre journées types.

    Ce nombre est imposé, non choisi par le modèle : la contrainte du programme
    d'optimisation est une égalité stricte sur le nombre de journées par saison.
    Trois est un compromis — à une seule journée optimisée, les journées
    extrêmes porteraient les trois quarts du poids annuel et le système serait
    dimensionné sur des situations atypiques.

SORTIES
    output/epm/pHours.csv          poids de chaque journée, en heures
    output/epm/pDemandProfile.csv  demande normalisée
    output/epm/pVREProfile.csv     facteurs de charge solaire et éolien
    output/representative_days_summary.csv   journées retenues et leur nature

AVERTISSEMENT
    `pDemandProfile` sort normalisé par le maximum NATIONAL, toutes zones
    confondues. EPM attend un maximum de 1 pour CHAQUE zone. La correction est
    faite à l'étape 5 par `check_outputs.py`, et non ici : la même série
    normalisée sert aussi à repérer la journée de pointe du système, et il faut
    alors conserver les tailles relatives des régions.
"""

import sys
from pathlib import Path

# .absolute() et non .resolve() : resolve() remonte la jonction jusqu au
# chemin reel, qui contient des espaces ("OneDrive - WBG"). Or le pipeline
# lance GAMS via subprocess sans proteger les chemins par des guillemets :
# le moindre espace casse la commande, GAMS n ecrit pas Results.gdx, et
# l erreur est masquee par stdout=DEVNULL. Toute la chaine doit donc etre
# lancee depuis un chemin sans espaces.
ICI = Path(__file__).absolute().parent
sys.path.insert(0, str(ICI.parent))

from representative_days.representativedays_pipeline import (  # noqa: E402
    run_representative_days_pipeline,
)

SAISONS = {m: (m - 1) // 3 + 1 for m in range(1, 13)}  # trimestres civils
ENTREES = {
    "Load": str(ICI / "input" / "load.csv"),
    "PV": str(ICI / "input" / "pv.csv"),
    "Wind": str(ICI / "input" / "wind.csv"),
}
SORTIE = ICI / "output"
N_JOURS = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3
N_CLUSTERS = 20
N_BINS = int(sys.argv[sys.argv.index("--bins") + 1]) if "--bins" in sys.argv else 10
SUFFIXE = f"_n{N_JOURS}_b{N_BINS}" if ("--n" in sys.argv or "--bins" in sys.argv) else ""
SORTIE = SORTIE.parent / (SORTIE.name + SUFFIXE)


def main():
    print("ÉTAPES 3 ET 4 — JOURS REPRÉSENTATIFS")
    print(f"   saisons    : 4 trimestres civils")
    print(f"   journées   : {N_JOURS} optimisées par saison, plus les extrêmes")
    print(f"   clusters   : {N_CLUSTERS}   bins : {N_BINS}")
    for k, v in ENTREES.items():
        print(f"   {k:<10} : {Path(v).name}")
    print()

    res = run_representative_days_pipeline(
        seasons_map=SAISONS,
        input_files=ENTREES,
        output_dir=SORTIE,
        epm_output_dir=SORTIE / "epm",
        summary_dir=SORTIE,
        n_representative_days=N_JOURS,
        n_clusters=N_CLUSTERS,
        n_bins=N_BINS,
        gams_main_file=str(ICI / "gams" / "OptimizationModelZone.gms"),
        verbose=True,
    )

    print()
    print("JOURNÉES RETENUES")
    rd = res.get("repr_days")
    if rd is not None:
        print(rd.to_string(index=False))
    sd = res.get("special_days")
    if sd is not None and len(sd):
        print()
        print("JOURNÉES EXTRÊMES")
        print(sd.to_string(index=False))
    return res


if __name__ == "__main__":
    main()
