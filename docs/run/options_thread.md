# Performance & Threads

EPM peut lancer plusieurs scénarios en parallèle avec le flag `--cpu`. Cette page explique comment dimensionner correctement ce paramètre selon votre machine.

---

## Concepts clés

| Terme | Signification |
|---|---|
| **Core** | Unité de calcul physique du processeur |
| **Thread** | Flux d'exécution ; avec hyperthreading, un core gère 2 threads |
| **vCPU** | En cloud/VM, 1 vCPU ≈ 1 thread |
| **`--cpu`** | Nombre de scénarios EPM lancés simultanément |
| **`threads`** | Nombre de threads alloués à chaque solve CPLEX, défini dans le fichier d'options du solveur |

> **Note** — Le flag `--cpu` contrôle en réalité le nombre de **jobs parallèles**, pas les CPUs directement. Ce nom sera renommé dans une prochaine version pour éviter la confusion.

---

## Les deux plafonds à respecter

Lancer `--cpu N` signifie que N scénarios tournent en même temps. Chaque scénario consomme de la RAM **et** des threads CPU. Il y a donc **deux plafonds indépendants** :

```
Plafond RAM  = RAM totale ÷ RAM par scénario
Plafond CPU  = vCPU totaux ÷ threads par scénario

--cpu = min(Plafond RAM, Plafond CPU)
```

Le plafond le plus bas est celui qui limite. Dépasser l'un ou l'autre provoque une contention des ressources et ralentit l'ensemble des jobs.

---

## Comment calculer en pratique

**Étape 1 — Connaître votre machine**

Notez la RAM totale et le nombre de vCPU disponibles.  
Sur Linux : `free -h` (RAM) et `nproc` (vCPU).

**Étape 2 — Mesurer la RAM par scénario**

Lancez un scénario seul et cherchez dans le fichier `.lst` ou la console GAMS Studio :
```
ProcTreeMemMonitor → VSS
```
C'est l'empreinte mémoire maximale de ce scénario. Utilisez cette valeur.

**Étape 3 — Connaître votre `threads`**

Regardez votre fichier d'options CPLEX (`cplex_baseline.opt`) :
```
threads = 8
```
Si la ligne est absente, CPLEX utilise tous les threads disponibles — à éviter en contexte parallèle.  
Voir [Options du solveur](options_solver.md) pour modifier cette valeur.

**Étape 4 — Calculer `--cpu`**

```
Plafond RAM  = RAM totale ÷ RAM par scénario
Plafond CPU  = vCPU totaux ÷ threads

--cpu = min(Plafond RAM, Plafond CPU)
```

---

## Exemple concret

Machine : **256 Go RAM, 32 vCPU**, scénario de ~32 Go, `threads = 8`

```
Plafond RAM  = 256 ÷ 32  = 8 jobs
Plafond CPU  = 32 ÷ 8    = 4 jobs

--cpu = min(8, 4) = 4
```

Ici le CPU est limitant. On lance avec `--cpu 4`, ce qui laisse ~64 Go de RAM inutilisée.

```sh
python epm.py --folder_input my_country --config config.csv --scenarios --cpu 4
```

---

## Arbitrage : threads vs. scénarios parallèles

Le nombre de threads par solve est un paramètre à ajuster selon votre usage.

**Beaucoup de scénarios à passer (longue file)**  
→ Préférez **moins de threads, plus de jobs parallèles**.  
La parallélisation entre scénarios est quasi parfaite (chaque job indépendant), alors que le gain en threads dans un solve est à rendement décroissant — passer de 4 à 8 threads accélère peu un solve donné. Davantage de solves en parallèle finissent une longue file plus rapidement.

*Exemple : baisser à `threads = 5` → Plafond CPU = 32 ÷ 5 = 6 jobs → `--cpu 6` au lieu de 4.*

**Peu de scénarios lourds (MIP sans `--simple`)**  
→ Préférez **plus de threads, moins de jobs parallèles**.  
Concentrez les ressources sur chaque solve pour le terminer plus vite.

> **Note** — Il est possible de dépasser légèrement le plafond CPU (ex. `--cpu 6` avec `threads = 8` sur 32 vCPU). L'OS partage alors le temps CPU entre les threads et les jobs s'exécutent plus lentement. Les résultats restent corrects mais le débit global est réduit par rapport à une allocation équilibrée.

---

## Récapitulatif

| Situation | Recommandation |
|---|---|
| Longue file de scénarios RMIP | Baisser `threads`, augmenter `--cpu` |
| Peu de scénarios MIP lourds | Garder `threads` élevé, `--cpu` plus bas |
| Machine partagée (2 modélisateurs) | Diviser `--cpu` par 2 |
