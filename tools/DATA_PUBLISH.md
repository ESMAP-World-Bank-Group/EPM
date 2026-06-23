# Publier / récupérer les données du modèle (store privé)

Les **données** d'EPM (input/output) ne vivent plus dans ce repo public : elles sont
dans un **store privé** (R2 aujourd'hui, S3 WB plus tard). Le repo ne garde que de
petits **pointeurs `.dvc`**. Ce dossier fournit l'outillage pour publier et récupérer.

> **Prototype.** Le store actuel est un bucket Cloudflare R2 (données de test).
> Données confidentielles → attendre le store S3 de la Banque.

---

## 🔧 Mise en place — une fois par machine

1. Installer les dépendances (inclut DVC) :
   ```
   pip install -r requirements.txt
   ```
2. Mettre tes clés d'accès au store :
   - copie `tools/.env.example` → `tools/.env`
   - colle les 4 valeurs (fournies par l'admin du store)
   - ⚠️ `tools/.env` est **gitignoré** : ne jamais committer de vraies clés.

C'est tout. La config du remote (URL/endpoint) est déjà dans le repo (`.dvc/config`).

---

## ⬆️ Publier (après avoir modifié des données) — Windows

**Double-clic sur `Publish.bat`** (à la racine du repo).

Ça fait tout : re-hash des données (`dvc add`) → commit + push des pointeurs →
`dvc push` (données → store, pour le serveur) → upload des copies lisibles
(inputs + `epm/output_view/`) → store, pour **EPM View**.

> Pour montrer des **résultats** dans EPM View : copie les runs voulus dans
> `epm/output_view/<run>/<scenario>/output_csv/` avant de publier (seuls les `.csv`
> sont envoyés). `epm/output_view/` est gitignoré (zone de staging locale).

---

## ⬇️ Récupérer code + données

- **Windows** : double-clic sur `Sync.bat`
- **Serveur Linux** : `bash tools/sync.sh`

= `git pull` (code + pointeurs) + `dvc pull` (données depuis le store).

---

## Onboarder un NOUVEAU modèle (une fois par modèle)

Sortir les données d'un modèle de git vers le store (la « bascule ») :
```
dvc init                                   # une fois par repo
dvc remote add -d store s3://<bucket>/dvcstore
dvc remote modify store endpointurl <endpoint>
# retirer la whitelist du dossier dans .gitignore, garder !epm/input/<data>.dvc
git rm -r --cached epm/input/<data_folder>
dvc add epm/input/<data_folder>
```
Puis publier (`Publish.bat`). Après ça, tout le monde fait juste *mise en place + publish/sync*.

---

## Côté EPM View (l'app)

EPM View lit les données par branche. Les branches dont les données sont dans le
store privé sont listées dans `R2_BRANCHES` (fichier `src/utils/epmFetch.js` du repo
`epm-data-explorer`). Pour brancher une nouvelle région : ajouter sa branche là.

---

## Fichiers de ce dossier (publication données)

- `publish.ps1` — moteur de `Publish.bat`
- `sync.ps1` / `sync.sh` — récupération (Windows / serveur)
- `upload_to_r2.py`, `upload_output_view_to_r2.py` — helpers d'upload lisible
- `.env.example` — modèle de clés (copier en `.env`, gitignoré)
