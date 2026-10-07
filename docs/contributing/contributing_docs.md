# Contribute Docs

Good documentation is as valuable as good code, and you can contribute to it too! Whether you spot a typo, want to clarify a section, or add missing content, all improvements are welcome. All documentation lives in the `docs/` folder as Markdown files. The navigation is defined in `mkdocs.yml` at the root of the repository.

---

## Edit an existing page

Open the relevant `.md` file in `docs/` and make your changes. No special tooling is required. Standard Markdown plus the [MkDocs Material extensions](https://squidfunk.github.io/mkdocs-material/) used in this project.

To preview locally:

```bash
pip install mkdocs-material
mkdocs serve
```

---

## Add a new page

1. Create a new `.md` file in the appropriate subfolder under `docs/`.
2. Add it to the `nav:` section of `mkdocs.yml`:

```yaml
nav:
  - Section:
    - Page title: section/filename.md
```

---

## Screenshots

The Dashboard is beta and its interface changes — screenshots go stale fast. When you add or
refresh one:

- capture a **local** instance, never a hosted demo (a deployment path such as
  `/opt/render/...` visible in the shot is a bug)
- use the `data_test` project and the `epm_env` environment, to match the rest of the docs
- keep a consistent window width (~1400 px)
- do not show controls the docs say are unavailable, such as the **Launch Run** button

Files go in `docs/images/dashboard/`.

---

## Git workflow

1. Make sure your `main` branch is up to date:
   ```bash
   git checkout main && git pull
   ```
2. Make your edits directly on `main` for documentation-only changes, or on a feature branch if the changes are substantial.
3. Commit and push:
   ```bash
   git add docs/
   git commit -m "Update documentation: ..."
   git push
   ```
4. Open a pull request on [GitHub](https://github.com/ESMAP-World-Bank-Group/EPM/pulls) with a short description of what you changed.

!!! tip
    If you maintain a separate fork or branch for a custom model, it's best to **manually copy** documentation changes back to `main` rather than merging, to avoid bringing in unrelated model-specific edits.
