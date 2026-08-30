# Rapport de résultats Black Sea 2026 — roadmap & layout

Générateur automatique d'un HTML de résultats (puis, plus tard, de slides) à partir
d'un dossier de run EPM. Cible immédiate : **Géorgie + Regional overview**, run
`simulations_run_20260819_204446`.

---

## 0. Ce que les données permettent (et ne permettent pas)

Inventaire fait sur le run du 19/08. 13 scénarios, 16 ans (2025-2040), 13 zones
internes + `iran_swap`, 28 blocs (4 saisons × 7 types de jour) × 24 h = 672 pas de temps.

### Sources par graphique

| Besoin | Fichier | Volume | Note |
|---|---|---|---|
| Capacité / génération annuelle par filière | `summary.csv` | 1,2 Mo, 9 648 l. | `Capacity: MW` et `Energy: GWh` ; `resolution` porte la filière |
| Imports / exports annuels | `summary.csv` | — | `Imports/Exports exchange: GWh`, plus `Annual Energy Imports/Exports External: GWh` (par partenaire) |
| Capacité de transmission | `summary.csv` + `pTransmissionMerged.csv` | — | `Transmission Capacity: MW`, par paire de zones |
| Échanges bilatéraux dirigés | `pTransmissionMerged.csv` (`Interchange`) | 172 ko/scénario | `Interchange[z][z2]` = flux **de z vers z2**, GWh/an → donne les flèches |
| Congestion | `pTransmissionMerged.csv` (`InterconUtilization`) | — | taux d'utilisation 0-1 par paire et par an |
| Dispatch horaire | `pDispatchComplete.csv` | **110 Mo/scénario** | `uni` ∈ {filières, `Demand`, `Imports`, `Exports`, `Storage Charge`, `Unmet demand`} |
| Coût marginal horaire | `pHourlyPrice.csv` | 7,4 Mo/scénario | $/MWh, par zone/h |
| Poids des blocs | `input/data_blacksea/pHours.csv` | — | pour l'axe x pondéré et les % d'heures |
| Lignes physiques + groupes projet | `pre-analysis/data/reference_lines.csv` | 53 lignes | colonne `project` = le groupe |
| NTC externe (Russie, Iran, Bulgarie…) | `input/data_blacksea/trade/pExtTransferLimit_*.csv` | — | **pas dans les sorties**, voir §0.2 |
| Géométrie | `zones.geojson`, `zones_ext.geojson`, `linestring_countries.geojson` | 450 ko | 8 voisins externes en polygones |

### 0.1 Cinq pièges identifiés — à traiter dans le générateur

1. **`NetImport` interne est faux d'un facteur 1000.** Georgia→AzerbaijanMain 2035 :
   `Interchange = 2915,7` mais `NetImport = −2,9157`. Les partenaires *externes*
   (Russie : 2604,9) sont, eux, cohérents avec `summary.csv`.
   → **on n'utilise jamais `NetImport` pour l'interne**, uniquement `Interchange`.
2. **La capacité des corridors externes n'existe pas dans les sorties.**
   `TransmissionCapacity` ne couvre que les paires internes (Géorgie : Armenia,
   AzerbaijanMain, EastAna — pas la Russie). Il faut lire
   `pExtTransferLimit_<variante>.csv` côté input, et donc résoudre
   scénario → fichier via `input_scenarios.csv`. C'est saisonnier et directionnel
   (`q`, `Import`/`Export`) : on prendra le max sur `q` par direction.
3. **`baseline` est le clone exact de `LC_Baseline`** (0 écart sur 9 648 lignes).
   → exclu de tous les sélecteurs.
4. **`CongestionShare` n'existe que pour 3 zones** (AzerbaijanMain, EastAna,
   Nakhchivan) → on affiche `InterconUtilization`, disponible partout.
5. **Pas de résultat par ligne physique.** EPM raisonne par *paire de zones*.
   Or plusieurs lignes réelles partagent une paire (TUR–GEO = Borçka–Akhaltsikhe
   HVDC 700 MW **+** Hopa–Batumi 220 kV hors service). Conséquence sur le graphe
   « une barre par ligne » : voir §2.1.5.

### 0.2 Volumétrie et cache

13 × 110 Mo de dispatch : hors de question de relire à chaque build. Étape de
pré-agrégation obligatoire :

```
tools/results_report/
  build.py          # CLI : --run, --countries, --scenarios, --out
  extract.py        # run EPM  ->  cache/<run>/<scenario>.json  (agrégats)
  charts.py         # séries -> SVG inline
  maps.py           # geojson + flux -> SVG inline
  render.py         # assemblage HTML + JS embarqué
  templates/report.css, report.js
  cache/            # gitignored
```

`extract.py` streame `pDispatchComplete.csv` en une passe (lecture par chunks
pandas), n'en garde que les années de dispatch demandées (2025 / 2030 / 2035),
agrège au niveau **pays** (somme des zones) et écrit du JSON compact. Ordre de
grandeur pour la Géorgie : 672 pas × ~12 séries × 3 ans × 2 scénarios ≈ 48 k
nombres ≈ 250 ko une fois arrondis à 1 décimale. Pour la Türkiye (8 zones
agrégées en 1) : même ordre.

### 0.3 Choix technique — ma recommandation

**HTML autonome, données JSON inlinées, rendu en JS vanilla (SVG), zéro CDN.**

- Même contrat que `calibration_review.html` : un fichier, ouvrable en `file:///`,
  envoyable par mail, pas de réseau.
- Mais **avec du JS** cette fois (le calibration review est du SVG statique écrit
  à la main). L'interactivité demandée — survol des flèches de la carte, tooltips
  de dispatch, bascule Baseline/Iso — n'est pas atteignable en SVG statique sans
  dupliquer chaque graphique par scénario.
- Pas de Chart.js ni de MapLibre : ~500 lignes de JS maison suffisent pour
  (a) l'aire empilée, (b) les barres empilées, (c) la carte projetée. Les
  charger depuis un CDN casserait l'usage hors-ligne, et les embarquer ferait
  +900 ko.
- **Carte** : projection équirectangulaire des geojson en chemins SVG, calculée
  en Python à la génération (géométries simplifiées Douglas-Peucker ~0,02°).
  Pas de fond de carte tuilé — un fond gris + frontières, comme les cartes du
  calibration review. Avantage : totalement hors-ligne et 100 % contrôlable.

Budget cible : **< 4 Mo** pour Géorgie + Regional. Si ça dépasse, on bascule le
dispatch sur un jour moyen par saison (28 → 4 blocs) au lieu des 28.

---

## I. Baseline vs Iso

Bandeau global en tête : sélecteur de scénario (Baseline / Iso / Δ), sélecteur
de langue FR-EN (même mécanique `.lf`/`.le` que le calibration review), et rappel
du run + date.

### I.a Par pays

#### I.a.1 → Géorgie *(premier livrable)*

**§1 — Capacité et génération annuelles, Baseline vs Iso**

Deux graphiques côte à côte, 2025→2040, un groupe de deux barres par année
(Baseline | Iso), empilement par filière.

- Gauche — **Capacité (GW)**. Empilement des filières + **au-dessus, en hachures
  diagonales, la capacité d'interconnexion** (somme des `Transmission Capacity`
  internes + NTC externes), sur le même axe MW.
- Droite — **Génération (TWh)**. Empilement des filières + **imports en hachures
  au-dessus de zéro** et **exports en hachures sous zéro**.
- Hachures : `<pattern>` SVG, 45°, couleur = celle du partenaire dominant,
  opacité 0,55, pour qu'on lise « ce n'est pas de la production domestique ».
- Survol : valeur, part du total, écart Baseline↔Iso.

**§2 — Comparaison avec le NDP (Baseline uniquement)**

Réutilisation directe du template de `calibration_review.html` §5 : trois barres
par année (Plan | Model | Δ), capacité et génération. Le code de
`ndp_build.py` est déjà écrit et validé — on le porte dans `charts.py`.
Sous le graphique, les panneaux Δ triés par |Δ| décroissant.
*(Pour la Géorgie le plan = pipeline hydro GSE ; les chiffres 2035/2040 sont
déjà à jour dans `_ndp_cmp_data.json`.)*

**§3 — Dispatch horaire — 2025, 2030, 2035**

Un graphique par année (3 empilés verticalement), convention
`epm-data-explorer/src/utils/dispatchSeries.js` reprise à l'identique :

- Aire empilée des filières (couleurs `TECHFUEL_COLORS` de l'explorer).
- `Imports` et `Storage Charge` dans la pile — `Exports` et `Storage Charge`
  **sous zéro** (EPM les écrit en positif dans `pDispatchComplete`, on inverse).
- `Unmet demand` en rouge vif au sommet.
- **Ligne demande** `#CC0000`, épaisseur 1,5.
- **Ligne coût marginal** sur axe droit ($/MWh), depuis `pHourlyPrice.csv`.
- Axe x groupé : 4 saisons × 7 types de jour, séparateurs verticaux (trait plein
  entre saisons, pointillé entre types de jour), et **part de l'année en % sous
  chaque bloc** (depuis `pHours.csv`).
- Sélecteur au-dessus : `Année pleine | Q1 | Q2 | Q3 | Q4` et
  `tous les jours | jour moyen`.

**§4 — Évolution des échanges**

Barres empilées 2025→2040 : imports (+) / exports (−) **par partenaire**
(AzerbaijanMain, EastAna, Armenia, Russia, Romania…), une couleur par partenaire,
avec un **marqueur « net »** (point) par année, comme `buildTrade()` de l'explorer.
Deux panneaux : Baseline et Iso, ou un panneau Δ selon le sélecteur.

**§5 — Cartes des flux — 2026, 2030, 2035**

Trois cartes côte à côte, centrées sur la Géorgie + ses voisins.

- Zones internes remplies (palette froide), zones externes en gris pointillé.
- **Flèches** entre centroïdes : largeur ∝ GWh échangés, orientation = sens net,
  couleur = taux d'utilisation (`InterconUtilization`) sur un dégradé
  vert → orange → rouge, la ligne saturée (>90 %) en rouge avec un liseré.
- Flèches vers les **zones externes** incluses (Russie, Roumanie…), pointant vers
  le centroïde du polygone voisin.
- **Survol** : partenaire, GWh import, GWh export, net, NTC (MW), utilisation %.
- Étiquette permanente du net en GWh sur les 3 plus gros corridors.

**§6 — Capacité par ligne, groupée par projet**

Barres groupées : un groupe par **projet** (`reference_lines.csv:project` —
BSTN, CTN, EWTC, GECO, BSSC, Zangezur, Trans-Caspian, Mid-Continental East,
« lignes existantes »), une barre par corridor dans le groupe, et **4 sous-barres
par corridor** (2025 / 2030 / 2035 / 2040) pour lire l'évolution.

> **Limite à trancher (§0.1 point 5)** : EPM ne renvoie pas la capacité
> ligne par ligne, seulement par paire de zones. Je propose donc **une barre par
> paire de zones**, et le détail des lignes physiques qui la composent (nom de
> poste, kV, statut, année d'entrée) dans le tooltip. C'est honnête et lisible ;
> une barre « par ligne » serait une répartition inventée.

#### I.a.2 … I.a.4 — Türkiye, Azerbaïdjan, Arménie

Strictement le même gabarit, paramétré par pays. Deux spécificités :

- **Türkiye** : 8 zones. Les graphiques annuels et le dispatch sont agrégés au
  niveau pays, mais la carte garde les 8 zones et montre aussi les **flux
  internes** (WestAna↔CenterAna…), qui sont l'essentiel du volume.
- **Azerbaïdjan** : Nakhchivan est une zone séparée et enclavée — la carte doit
  la traiter comme telle (déjà fait dans `zones.geojson`, cf. commit `f5ad8c24`).

### I.b Regional overview *(deuxième livrable)*

**§1 — Capacité et génération régionales**

- Empilement par **pays** (4 couleurs) puis, en second graphique, par **filière**
  pour toute la région. Baseline vs Iso côte à côte.
- Bandeau de chiffres clés au-dessus : capacité installée, génération, échanges
  intra-région, échanges avec l'extérieur, émissions, coût NPV — pour chaque
  scénario, avec le Δ.

**§2 — Cartes régionales des flux — 2026, 2030, 2040**

Trois cartes **en colonne à gauche (≈ 62 % de la largeur)**, et **en face à
droite les key findings** rédigés en regard de chaque carte, dans un encadré :

```
┌───────────────────────────┬──────────────────────┐
│  Carte 2026               │  Key findings 2026   │
│  (flux + congestion)      │  • …                 │
├───────────────────────────┼──────────────────────┤
│  Carte 2030               │  Key findings 2030   │
├───────────────────────────┼──────────────────────┤
│  Carte 2040               │  Key findings 2040   │
└───────────────────────────┴──────────────────────┘
```

Les findings sont **calculés puis rédigés** : le générateur produit les faits
(corridor le plus chargé, corridors saturés, plus gros basculement de sens entre
deux années, dépendance externe max) et le texte narratif est écrit par-dessus,
comme dans le calibration review. Chaque chiffre cité est un chiffre extrait,
jamais saisi à la main → pas de dérive quand le run change.

**§3 — Imports / exports régionaux**

- Matrice origine-destination (heatmap 14×14, GWh) par année, avec sélecteur
  d'année.
- Barres imports/exports par pays, et **balance avec l'extérieur de la région**
  (Russie, Iran, Bulgarie, Grèce, Roumanie, Kazakhstan) séparée de l'échange
  intra-région — c'est la lecture qui compte pour le RETRADE.
- Courbe du taux d'utilisation moyen des corridors, 2025→2040, par scénario.

---

## II. Ordre de construction

| Étape | Contenu | Validation |
|---|---|---|
| **1** | `extract.py` + cache pour LC_Baseline et LC_Iso, Géorgie seule | les totaux du cache = `summary.csv` à 0,1 % près |
| **2** | I.a.1 §1, §2, §4 (annuel + NDP + échanges) | comparaison manuelle avec `summary.csv` |
| **3** | I.a.1 §3 (dispatch) | somme pondérée par `pHours` = `Energy: GWh` annuel |
| **4** | I.a.1 §5, §6 (carte + lignes) | flux entrants = flux sortants par corridor |
| **5** | **Revue avec toi** | ← on s'arrête ici avant d'aller plus loin |
| **6** | I.b complet (Regional) | |
| **7** | **Revue** | |
| **8** | Türkiye, Azerbaïdjan, Arménie (même gabarit) | |
| **9** | Export slides (`python-pptx`) à partir des mêmes séries | |

---

## III. Points à trancher

1. **Périmètre des scénarios.** L'outline ne couvre que Baseline vs Iso. Les 11
   autres (BSTN, CTN, EWTC, GECO, BSSC, Zangezur, TransCaspian, 60pct,
   AllProjects, FreeExp) : sections II et III plus tard, ou un sélecteur global
   qui rend n'importe quelle paire comparable dès maintenant ? *Je recommande le
   sélecteur* — le coût marginal est nul une fois le gabarit paramétré.
2. **Années des cartes.** Tu dis 2026/2030/2035 pour les pays et 2026/2030/2040
   pour le régional. Volontaire, ou on harmonise sur 2026/2030/2040 partout
   (2035 étant déjà couvert par le dispatch) ?
3. **Barres par ligne vs par corridor** — cf. §I.a.1 §6.
4. **Langue.** Bilingue FR/EN comme le calibration review, ou EN seul ? Le
   bilingue double le travail de rédaction des key findings.
5. **Où le fichier vit.** Proposition : sortie dans
   `blacksea_2026/Data/results/results_review.html`, code dans
   `EPM/tools/results_report/`, cache gitignoré.
