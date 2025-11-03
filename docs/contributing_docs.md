# Contributing to the documentation

All markdown files for the documentation are in the `docs/` folder. To contribute:

- You can improve an existing page by editing its `.md` file. For example, update `docs/model_formulation.md` to improve the model formulation section.
- You can add a new page by creating a new markdown file in `docs/`. If you do this, don’t forget to:
  1. Add the new file to the `_toc.yml` file so it appears in the documentation sidebar.

## How to update and improve the documentation


When contributing to the EPM documentation, please follow these steps to keep your changes compatible with the latest version of the project.

### 1 Recommended setup
It’s best to have **two local EPM folders**:
- One folder tracking the **`main`** branch (the official version).
- Another folder for your **custom developments or models**.

This setup helps you avoid merge conflicts when you want to contribute only documentation updates.

### 2 Update your local `main` branch
Before making any changes to the documentation, make sure you are on the main branch and that it’s up to date:

```bash
# Go to your EPM directory that tracks the main branch
cd path/to/EPM

# Switch to the main branch
git checkout main

# Pull the latest changes from the remote repository
git pull
```

### 3 Make your documentation changes
Modify the documentation files directly in the `main` branch (e.g., under `docs/`, `README.md`, etc.).

1. Go to the `docs/` folder.
2. Edit an existing markdown file or create a new one.
3. If you add a new file, update `_toc.yml` to include it in the navigation.
4. Add your changes to git and push them to the `main` branch.

If you already made documentation changes in another branch (your custom branch),
it’s usually best to **manually copy or reapply** those changes onto the updated `main` branch,
rather than merging that branch — this avoids bringing in unrelated code.

### 4 Commit your changes

```bash
git add path/to/modified/files
git commit -m "Update documentation on ..."
```

### 5 Create a Pull Request (PR)
Push your changes and open a PR to merge them into `main`.

```bash
git push origin main
```

Then:
1. Open the repository on [GitHub](https://github.com/ESMAP-World-Bank-Group/EPM/pulls).
2. Click **“Compare & pull request”**.
3. Add a clear title and a short explanation of what you changed or improved.
4. Submit your PR for review.

---

**Summary**
- Keep a separate EPM folder for `main` and another for your personal development.
- Always `git pull` before editing documentation.
- Apply edits directly to the updated `main`.
- Create a Pull Request when ready.



Thank you for helping improve this project ! 
