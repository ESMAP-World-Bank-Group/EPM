# Resolution Advisor — Méthodologie détaillée

Ce document explique pas à pas **ce que fait le code**, **pourquoi**, et **comment chaque fichier contribue au résultat final**.

---

## La question centrale

Avant de construire un modèle EPM, il faut répondre à deux questions :

> **Combien de zones** faut-il représenter pour que le modèle soit physiquement crédible ?
> **Combien de jours représentatifs** faut-il pour que la chronologie soit bien capturée ?

Trop peu de zones → on manque des contraintes de réseau réelles, les prix et le dispatch sont faux.
Trop de zones → le modèle devient trop lourd, les temps de résolution explosent.
Même raisonnement pour les jours représentatifs.

Le Resolution Advisor calcule un **plancher** (ce que la physique exige au minimum) et un **plafond** (ce que le budget de calcul permet au maximum), puis propose des points de test entre les deux.

---

## Les deux modes

### Mode manuel (`--config`)
Tu fournis toi-même les paramètres dans un fichier YAML (`config/blacksea.yaml`).
Le code fait juste les calculs à partir de ce que tu as mis.

### Mode auto (`--auto`)
Le code va **chercher les données lui-même** depuis internet (OSM, Natural Earth, GPPD),
puis calcule les paramètres automatiquement avant de faire les mêmes calculs.

---

## Exécution pas à pas (mode auto)

Quand tu lances :
```bash
python advise.py --countries TUR ROU BGR --auto
```

Voici exactement ce qui se passe, dans l'ordre :

---

### Etape 1 — Chargement des données géographiques
**Fichier : `fetch/natural_earth.py`**

Charge deux jeux de données depuis [Natural Earth](https://www.naturalearthdata.com/) :

**Frontières pays** (`ne_110m_admin_0_countries.shp`)
- Polygones des frontières de chaque pays
- Utilisés pour : calculer l'aire du pays, tracer les bounding boxes, clipper les géométries
- Résolution 110m (suffisant pour notre usage)

**Villes peuplées** (`ne_110m_populated_places.shp`)
- Toutes les villes avec leur population estimée
- Utilisés pour : identifier les centres de charge (load centers), calculer leur dispersion géographique

Les fichiers sont d'abord cherchés dans `dataset/maps/` (s'ils existent dans le repo),
sinon téléchargés depuis naciscdn.org et mis en cache dans `cache/natural_earth/`.
Les téléchargements suivants utilisent le cache — pas de re-fetch.

---

### Etape 2 — Chargement des centrales
**Fichier : `fetch/gppd.py`**

Charge la **Global Power Plant Database** (WRI) — ~35 000 centrales mondiales avec :
localisation (lat/lon), combustible, capacité installée (MW), année de mise en service.

Utilisé pour : identifier si l'hydroélectricité est concentrée loin des centres de charge
(ce qui justifie une zone supplémentaire pour représenter le corridor de transport).

> **Note** : Les URLs WRI sont actuellement mortes (404). Si les données ne se téléchargent pas,
> les paramètres hydro sont ignorés (le reste fonctionne normalement).
> Pour activer : télécharger manuellement depuis https://datasets.wri.org/dataset/globalpowerplantdatabase
> et placer à `cache/gppd/global_power_plant_database.csv`.

---

### Etape 3 — Collecte des données réseau OSM
**Fichier : `fetch/osm.py`**

Interroge l'**API Overpass** d'OpenStreetMap pour récupérer l'infrastructure électrique HT :

**Substations** (`power=substation`) :
- Tous les postes de transformation dans le bounding box du pays
- Attributs récupérés : position (lat/lon), tension (kV), nom

**Lignes HT** (`power=line` ou `power=cable` avec attribut `voltage`) :
- Toutes les lignes de transport >= 100 kV
- Attributs : géométrie complète (liste de coordonnées), tension (kV)

Les requêtes sont **mises en cache** (hash MD5 de la requête → fichier JSON dans `cache/osm/`).
Un délai de 3 secondes est respecté entre les requêtes pour ne pas surcharger l'API.
En cas d'échec (proxy corporate, API indisponible), retourne une liste vide — les étapes suivantes
fonctionnent en mode dégradé.

---

### Etape 4 — Calcul des paramètres par pays
**Fichier : `auto.py`** (orchestration) + fichiers `compute/`

Pour chaque pays, `auto.py` appelle 5 fonctions de calcul :

#### 4a. Aire du pays
**Fichier : `compute/area.py`**

```
boundaries_gdf  ->  reprojection EPSG:6933 (equal-area)  ->  area_m2  ->  area_km2
```

Utilise la **projection equal-area cylindrique** (EPSG:6933) pour que les polygones lat/lon
soient convertis en mètres avant le calcul d'aire. Sans cette reprojection, les degrés en
latitude et longitude ne correspondent pas à la même distance réelle (1° de longitude à
l'équateur = 111 km, mais à 45°N = 78 km).

Fallback si la reprojection échoue : approximation sphérique depuis la bounding box
(`lat_span × 111 km × lon_span × 111 km × cos(lat_moyenne)`).

**Pourquoi c'est utile :** un grand pays (> 500 000 km²) a statistiquement plus de diversité
de ressources et de distance réseau — il mérite au moins une zone supplémentaire.

---

#### 4b. Hétérogénéité des ressources RE
**Fichier : `compute/re_spread.py`**

Calcule un **proxy géographique** de la variabilité du facteur de capacité RE à travers le pays.

```
lat_span = max_latitude - min_latitude  (en degrés)
lon_span = max_longitude - min_longitude
spread = lat_span × 0.022 + lon_span × 0.008 × 0.5
spread = min(spread, 0.50)  # plafonné à 50%
```

L'idée : l'étendue nord-sud capture les gradients d'irradiation solaire (plus fort au sud),
l'étendue est-ouest capture les gradients de vent (régimes différents côte vs intérieur).
Les coefficients (0.022 et 0.008) sont calibrés empiriquement pour rester cohérents
avec des études de CF variability en Europe et au Moyen-Orient.

C'est un **proxy sans données API** — pas de clé d'accès nécessaire, toujours disponible.
Pour plus de précision, il faudrait des données ERA5 ou MERRA-2.

**Seuil :** si spread > 0.25 (25%), une zone supplémentaire est justifiée (sinon les RE
moyennées par zone donnent un résultat biaisé).

---

#### 4c. Distance entre centres de charge
**Fichier : `compute/load_centers.py`**

Prend les **5 plus grandes villes** du pays (par population dans Natural Earth),
calcule toutes les distances paires avec la formule haversine, retourne la distance maximale.

```
Si max_distance > 350 km  ->  distant_load_centers = True
```

**Pourquoi 350 km ?** Au-delà de cette distance, une ligne HT de 220 kV transportant
~1000 MW subit des pertes d'environ 5-8%, et des contraintes de transit peuvent apparaître
avec une forte charge. C'est la limite empirique utilisée dans les études ENTSO-E.

**Pourquoi c'est utile :** si Istanbul et Ankara sont à 350 km (limite exacte dans ce cas),
modéliser la Turquie comme une seule zone uniforme suppose qu'une centrale à Istanbul peut
alimenter Ankara sans contrainte. Ce n'est pas vrai en période de pointe.

---

#### 4d. Concentration de l'hydroélectricité
**Fichier : `compute/hydro_concentration.py`**

Compare la position géographique des centrales hydro avec celle des centres de charge :
1. Calcule le **centroïde pondéré** des centrales hydro (pondéré par capacité MW)
2. Calcule le **centroïde pondéré** des villes (pondéré par population)
3. Mesure la distance haversine entre les deux centroïdes

```
Si distance > 150 km  ->  hydro_concentration = True  (hydro loin du load)
```

**Pourquoi c'est utile :** si toute l'hydro est dans les montagnes à l'est et toute la
demande est sur le littoral ouest, il y a forcément un corridor de transport critique
entre les deux. Ne pas le représenter explicitement dans le modèle introduit des biais
sur le dispatch hydro et les prix nodaux.

---

#### 4e. Corridors de congestion réseau
**Fichier : `compute/network_bottlenecks.py`**

C'est le calcul le plus sophistiqué. Il détecte les **goulots d'étranglement** dans le
réseau OSM en utilisant la théorie des graphes.

**Etape 1 — Construction du graphe**
- Chaque **substations** devient un noeud
- Chaque **ligne HT** devient une arête entre ses deux extrémités
- Les extrémités de lignes sont "snappées" à la sous-station la plus proche dans un rayon
  de 15 km (pour raccorder les lignes qui ne passent pas exactement par les postes dans OSM)

**Etape 2 — Edge betweenness centrality** (algorithme NetworkX)

Pour chaque paire de noeuds (A, B) dans le réseau, on calcule le plus court chemin.
La **betweenness centrality** d'une arête = fraction des plus courts chemins qui passent
par cette arête.

Une arête avec betweenness élevée est **critique** : beaucoup de flux "passent" par elle,
au sens topologique. Si elle était saturée, de nombreuses paires de noeuds seraient
déconnectées ou forcées par des chemins plus longs.

**Etape 3 — Identification des bottlenecks**
```
seuil = moyenne(betweenness) + 2 × écart-type(betweenness)
bottleneck_edges = arêtes avec betweenness > seuil
```

Le seuil `moyenne + 2σ` est une convention statistique standard pour identifier les valeurs
aberrantes dans une distribution — ici, les lignes qui se démarquent vraiment du réseau.

**Etape 4 — Comptage des corridors**
Les arêtes bottleneck adjacentes (partageant un noeud) sont regroupées en **composantes
connexes**. Chaque composante = un corridor de congestion distinct.

```
TUR : 80 arêtes bottleneck -> 5 corridors distincts
ROU : 61 arêtes bottleneck -> 2 corridors distincts
BGR : 38 arêtes bottleneck -> 1 corridor distinct
```

**Pourquoi c'est utile :** chaque corridor identifié est un endroit où le réseau peut être
saturé, ce qui justifie une frontière de zone. C'est exactement la logique PyPSA-Eur pour
la segmentation des réseaux européens.

---

### Etape 5 — Assemblage des CountryConfig
**Fichier : `auto.py`** + **`schema.py`**

Tous les paramètres calculés sont assemblés dans un objet `CountryConfig` :

```python
CountryConfig(
    name="TUR",
    area_km2=798647,
    n_bidding_zones=1,          # non automatisé, fixé à 1 par défaut
    known_congestion_splits=5,  # calculé par network_bottlenecks.py
    re_cf_spread=0.21,          # calculé par re_spread.py
    distant_load_centers=True,  # calculé par load_centers.py
    hydro_concentration=False,  # calculé par hydro_concentration.py (GPPD absent)
    data_quality="good",        # déduit de la couverture OSM
)
```

La `data_quality` est déduite automatiquement :
- `good` si OSM a > 500 substations et > 500 lignes pour le pays
- `medium` si OSM a 50-500 éléments
- `limited` si OSM a < 50 éléments ou si le fetch a échoué

---

### Etape 6 — Recommandation spatiale
**Fichier : `spatial/recommender.py`**

Pour chaque pays, calcule un **plancher physique** par accumulation de drivers :

| Driver | Contribution |
|--------|-------------|
| Bidding zones officielles | = n_bidding_zones |
| Corridors de congestion | +n corridors |
| RE spread > 25% | +1 zone |
| Aire > 500 000 km² | +1 zone |
| Load centers > 350 km | +1 zone |
| Hydro loin du load | +1 zone |

Exemple pour TUR : 1 (base) + 5 (congestion) + 0 (RE spread 21% < 25%) + 1 (grande aire) + 1 (load centers) = **8 zones**

Ce plancher est ensuite **capé par la qualité des données** :
- `good` → max 6 zones (on ne peut pas décomposer plus que ce qu'on peut calibrer)
- `medium` → max 4 zones
- `limited` → max 1 zone

La logique : même si le réseau physique justifie 8 zones, si on n'a que des données nationales
agrégées, les paramètres d'une zone 6 seraient inventés. Mieux vaut 6 zones bien calibrées
que 8 zones dont 2 sont des fictions.

**Plafond** de calcul :
```
ceiling = variable_budget / (N_heures_repr × N_années × N_scénarios)
        = 8 000 000 / (384h × 3ans × 1scénario)
        = 30 zones
```

Le `variable_budget` de 8M est une estimation du nombre de variables LP/MIP qu'un solveur
CPLEX peut traiter en < 6h sur 64 GB RAM, basé sur des benchmarks EPM.

---

### Etape 7 — Recommandation temporelle
**Fichier : `temporal/recommender.py`**

Calcule le nombre minimum de **jours représentatifs** (pas les vrais jours de l'année —
des "types de jours" agrégés qui représentent la chronologie annuelle).

**Plancher** (règles cumulatives) :
- Baseline : 4 jours (1 par saison)
- RE >= 20% → min 8 jours (capturer la variabilité hebdomadaire du vent)
- RE >= 35% → min 12 jours (capturer les coïncidences faible-RE/forte-demande)
- RE >= 50% → min 16 jours
- Storage `medium` → +2 jours (cycles de charge/décharge multi-jours)
- Storage `high` → +4 jours
- Hydro saisonnier fort → min 8 jours (saisons sèche/humide séparées)

**Jours extrêmes** (toujours ajoutés par-dessus) :
- 2 jours minimum : pic de demande + événement min-RE
- Si RE >= 30% : 3 jours (ajoute sécheresse de vent)

Ces jours extrêmes sont distincts des jours représentatifs : ils ne représentent pas
la chronologie typique mais les pires cas qui dimensionnent les capacités de backup.

**Plafond** :
```
max_days = variable_budget / (N_zones × 24h × N_années × N_scénarios)
         = 8 000 000 / (11 zones × 24h × 3ans × 1scénario)
         = 36 jours
```

---

### Etape 8 — Affichage et sauvegarde
**Fichier : `advise.py`**

Assemble les résultats en un tableau formaté (ou JSON avec `--output json`).
Avec `--save`, sauvegarde un JSON dans `output/`.

---

## Résumé : qui calcule quoi

```
advise.py                 CLI, orchestre tout, affiche le résultat
auto.py                   Orchestre la collecte + calcul en mode auto
schema.py                 Structures de données (CountryConfig, AdvisorConfig, ...)

fetch/
  natural_earth.py        Telecharge/charge frontières + villes (Natural Earth)
  osm.py                  Requête Overpass API -> substations + lignes HT
  gppd.py                 Telecharge/charge GPPD -> centrales

compute/
  area.py                 aire_km2 = reprojection EPSG:6933 -> calcul géométrique
  re_spread.py            re_spread = proxy géographique lat/lon
  load_centers.py         distant_load_centers = max distance entre villes > 350 km ?
  hydro_concentration.py  hydro_concentration = centroïde hydro vs centroïde load > 150 km ?
  network_bottlenecks.py  known_congestion_splits = edge betweenness sur graphe OSM

spatial/
  recommender.py          plancher + plafond + candidats (nb zones)

temporal/
  recommender.py          plancher + plafond + candidats (nb jours représentatifs)
```

---

## Ce que le code ne fait PAS (encore)

- Il ne **génère pas les zones** — il dit combien il en faut. Pour les générer, voir `pipelines/zone_pipeline.py`.
- Il ne **sélectionne pas les jours représentatifs** — pour ça, voir `representative_days/` et le pipeline tsam/Poncelet.
- Le paramètre `n_bidding_zones` n'est pas automatisé (toujours 1 par défaut en mode auto) — à renseigner manuellement dans le YAML si le pays a des marchés zonaux officiels.
- La `data_quality` est déduite de la densité OSM, pas de la disponibilité des données EPM réelles (load horaire zonal, etc.).
