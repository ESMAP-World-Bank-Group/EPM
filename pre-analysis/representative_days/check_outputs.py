"""Étape 5 — Correction et contrôle des sorties destinées à EPM.

OBJET
    Corriger le seul défaut connu du pipeline, puis vérifier que les trois
    fichiers livrés à EPM sont cohérents entre eux et avec les données d'origine.

LA CORRECTION — normalisation de pDemandProfile par zone
    `utils.py`, fonction `_ensure_normalized_series`, divise la série de demande
    par son maximum GLOBAL, toutes zones confondues. EPM attend un profil dont
    le maximum vaut 1 pour CHAQUE zone, puisqu'il le multiplie ensuite par la
    pointe zonale de `pDemandForecast`.

    Sans correction, et d'après les pointes mesurées de 2025, OCC atteindrait
    1,00 mais SIBCS plafonnerait à 0,058 : le modèle croirait que la Basse-
    Californie du Sud n'appelle jamais plus du vingtième de sa puissance.
    Le rapport entre la plus grande et la plus petite zone est de 17.

    La correction est faite ICI et non en amont : la même série normalisée sert
    aussi à repérer la journée de pointe du système, et il faut alors conserver
    les tailles relatives des régions. Corriger à l'entrée réglerait un problème
    et en créerait un autre.

LES CONTRÔLES
    1. somme de pHours égale à 8 760 heures
    2. maximum de pDemandProfile égal à 1 pour chacune des 9 zones
    3. zones et journées cohérentes entre les trois fichiers
    4. énergie annuelle reconstituée, comparée à la mesure du CENACE
    5. facteurs de charge solaire et éolien reconstitués, comparés aux séries
       horaires d'origine

    Le contrôle 4 est le plus parlant : il reconstitue l'énergie annuelle de
    chaque GCR à partir des seules journées types et de leurs poids, puis la
    confronte aux 8 760 heures réelles. Un écart important signalerait que les
    journées retenues ne représentent pas l'année.

SORTIE
    output/epm/pDemandProfile.csv corrigé sur place, l'original étant conservé
    sous pDemandProfile_brut.csv.
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
    """Poids en heures de chaque couple saison / type de jour."""
    t = [c for c in ph.columns if c.startswith("t")]
    cle = [c for c in ph.columns if c not in t][:2]
    return ph.set_index(cle)[t].sum(axis=1)


def corriger(verbeux: bool = True) -> pd.DataFrame:
    """Renormalise pDemandProfile zone par zone."""
    f = EPM / "pDemandProfile.csv"
    brut = EPM / "pDemandProfile_brut.csv"
    shutil.copy2(f, brut)   # idem : re-copie a chaque execution, correction idempotente
    d = pd.read_csv(brut)
    t = [c for c in d.columns if c.startswith("t")]
    zc = d.columns[0]
    for c in t:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if verbeux:
        print("CORRECTION — normalisation de pDemandProfile par zone")
        print(f"   {'zone':<8}{'max avant':>12}{'max après':>12}")
        print("   " + "-" * 32)
    for z, g in d.groupby(zc):
        m = g[t].max().max()
        if m and m > 0:
            d.loc[d[zc] == z, t] = (g[t] / m).values
        if verbeux:
            print(f"   {z:<8}{m:>12.4f}{1.0:>12.4f}")
    d.to_csv(f, index=False)
    if verbeux:
        print(f"   original conservé sous {brut.name}")
        print()
    return d


JOURS_SAISON = {"Q1": 90, "Q2": 91, "Q3": 92, "Q4": 92}   # 2025, annee non bissextile


def corriger_pHours(verbeux: bool = True) -> pd.DataFrame:
    """Ramene les poids de chaque saison au nombre reel de jours du calendrier.

    DEFAUT CORRIGE
        Le pipeline attribue a chaque journee extreme le poids de son groupe de
        classification, mais ne retire pas ce meme nombre de jours du lot soumis
        a l optimisation. Les deux comptes ne se rejoignent donc pas, et la somme
        de pHours s ecarte de 8 760 heures.
        Avec un jeu de donnees, les groupes peuvent tomber juste par hasard et le
        defaut reste invisible ; il ne l est plus des que les donnees changent.
        Constate ici : Q4 pesait 107 jours au lieu de 92.

    CORRECTION
        Renormalisation PAR SAISON, proportionnelle. Les poids relatifs entre
        journees d une meme saison sont conserves, et chaque saison retrouve son
        nombre reel de jours. C est la correction la plus neutre : elle ne
        privilegie ni les journees extremes ni les journees optimisees.
    """
    f = EPM / "pHours.csv"
    brut = EPM / "pHours_brut.csv"
    # On re-copie a CHAQUE execution : une sauvegarde conservee d une execution
    # precedente ferait corriger des poids perimes apres une nouvelle selection.
    # L operation est idempotente -- renormaliser un fichier deja correct donne
    # un facteur de 1 -- donc relancer le controle seul reste sans effet.
    shutil.copy2(f, brut)
    ph = pd.read_csv(brut)
    t = [c for c in ph.columns if c.startswith("t")]
    cs, cd = [c for c in ph.columns if c not in t][:2]
    if verbeux:
        print("CORRECTION - poids des journees par saison")
        print(f"   {'saison':<8}{'avant':>9}{'reel':>8}{'facteur':>10}")
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
        print(f"   total {ph[t].sum().sum():.0f} h   original conserve sous {brut.name}")
        print()
    return ph


def controler(verbeux: bool = True) -> bool:
    ok = True
    ph = pd.read_csv(EPM / "pHours.csv")
    dp = pd.read_csv(EPM / "pDemandProfile.csv")
    vp = pd.read_csv(EPM / "pVREProfile.csv")
    w = _heures(ph)

    print("CONTRÔLES")
    # 1 — somme des heures
    total = w.sum()
    bon = abs(total - 8760) < 0.5
    ok &= bon
    print(f"   1. somme de pHours            {total:>10.0f} h   attendu 8 760   "
          f"{'ok' if bon else 'ANOMALIE'}")
    print(f"      {len(ph)} journées types, soit {total / 24:.0f} jours")

    # 2 — maximum par zone
    t = [c for c in dp.columns if HEURE.match(c)]
    zc = dp.columns[0]
    for c in t:
        dp[c] = pd.to_numeric(dp[c], errors="coerce")
    mx = dp.groupby(zc)[t].max().max(axis=1)
    bon = bool(((mx - 1).abs() < 1e-6).all())
    ok &= bon
    print(f"   2. max de pDemandProfile      {mx.min():.4f} à {mx.max():.4f}      "
          f"{'ok, 1,00 partout' if bon else 'ANOMALIE'}")

    # 3 — coherence des zones
    zd, zv = set(dp[zc]), set(vp[vp.columns[0]])
    bon = zd == set(ZONES) and zv == set(ZONES)
    ok &= bon
    print(f"   3. zones                      demande {len(zd)}, VRE {len(zv)}      "
          f"{'ok' if bon else 'ANOMALIE : ' + str(set(ZONES) ^ zd)}")

    # 4 — energie annuelle reconstituee
    print()
    print("   4. énergie annuelle reconstituée, contre la mesure CENACE")
    ld = pd.read_csv(ENTREE / "load.csv")
    reel = ld.groupby("zone").value.sum() / 1000  # GWh
    pointe = ld.groupby("zone").value.max()
    cle = [c for c in dp.columns if c not in t][:3]
    print(f"      {'zone':<8}{'reconstitué':>13}{'mesuré':>11}{'écart':>9}")
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
    print(f"      écart maximal {pire:.1f} %   {'ok' if bon else 'A EXAMINER'}")

    # 5 — facteurs de charge VRE reconstitues
    print()
    print("   5. facteur de charge VRE reconstitué, contre la série horaire")
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
        print(f"      {tech:<6}{100 * num / den:>7.1f}%   série horaire "
              f"{100 * src.value.mean():>6.1f}%")

    print()
    print(f"   -> {'TOUS LES CONTRÔLES PASSENT' if ok else 'ANOMALIES, voir ci-dessus'}")
    return ok


# --------------------------------------------------------------------------- #
# Export au format attendu par EPM
# --------------------------------------------------------------------------- #

MODELE = ICI.parent.parent / "epm" / "input" / "data_mexico_2026"
EXPORT = ICI / "output" / "epm_modele"

# Le pipeline et le modele ne nomment pas les colonnes de la meme facon.
RENOMMAGE = {"zone": "z", "season": "q", "daytype": "d", "fuel": "tech"}
# Les colonnes horaires sont t1 a t24 ; un simple startswith("t") attraperait
# aussi la colonne "tech" et la dupliquerait dans la sortie.
HEURE = re.compile(r"^t\d+$")
# Ni les technologies.
TECHNOS = {"Wind": "OnshoreWind", "PV": "PV"}


def exporter(verbeux: bool = True) -> dict:
    """Met les trois sorties au format du modele et les ecrit dans output/epm_modele.

    TROIS TRANSPOSITIONS

    1. Noms de colonnes. Le pipeline ecrit `zone, season, daytype, fuel` ;
       le modele attend `z, q, d, tech`.

    2. Nom de technologie. Le pipeline ecrit `Wind`, le modele `OnshoreWind`,
       qui est la valeur de la colonne `tech` dans pGenDataInput.

    3. Profil de la centrale solaire a concentration. main.gms ligne 662 pose
           pCSPProfile(cs,q,d,t) = sum(..., pVREProfile(z,tech,q,d,t))
       Le profil CSP est donc lu dans pVREProfile sous la technologie
       `CSPPlant`. Le pipeline ne produit que PV et eolien : on ajoute une
       ligne CSPPlant pour SIBCS, seule zone portant une centrale CSP, en
       reprenant le profil photovoltaique de cette zone.

       C'est une APPROXIMATION declaree. La ressource d'une centrale a
       concentration est l'irradiance directe normale, quand le photovoltaique
       repond a l'irradiance globale inclinee. Les deux sont fortement
       correlees sous ciel clair, et le suiveur a un axe retenu pour le PV
       rapproche encore les deux. L'ecart porte surtout sur les journees
       nuageuses, ou la CSP produit moins que ne le laisse croire ce profil.
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
        print("EXPORT AU FORMAT DU MODELE")
    for nom, d in sorties.items():
        d.to_csv(EXPORT / nom, index=False, encoding="utf-8-sig")
        if verbeux:
            print(f"   {nom:<22}{len(d):>5} lignes   colonnes {', '.join(d.columns[:4])}")
    if verbeux:
        print(f"   ecrit dans {EXPORT.relative_to(ICI)}")
        print()
        print("   a copier vers le modele :")
        print(f"      pHours.csv          -> {MODELE.name}/")
        print(f"      pDemandProfile.csv  -> {MODELE.name}/load/")
        print(f"      pVREProfile.csv     -> {MODELE.name}/supply/")
    return sorties


def main():
    if not (EPM / "pHours.csv").exists():
        raise FileNotFoundError(f"Sorties EPM introuvables dans {EPM}")
    print("ÉTAPE 5 — CORRECTION ET CONTRÔLE")
    print()
    corriger_pHours()
    corriger()
    ok = controler()
    exporter()
    resume = ICI / "output" / "representative_days_summary.csv"
    if resume.exists():
        print()
        print("JOURNÉES RETENUES")
        print(pd.read_csv(resume).to_string(index=False))
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as exc:  # pragma: no cover
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        raise
