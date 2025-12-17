# Git Procedure for Beginners

Before asking for help, please first search the issue on Google or ask an AI assistant. If the problem persists, then ask a colleague.

## Introduction

Git is a **version control system** used to manage code changes. A key concept in Git is the **branch**.

### What is a branch?
A **branch** is a copy of the code where you can work independently without affecting the main version. This allows multiple people to work in parallel.

Two types of branches:
- **Local**: Exists only on your computer.
- **Remote**: Shared on GitHub (or another server, but here GitHub).

You’ll often need to:
- Merge updates from `main` into your branch
- Push your own changes to `main`

You only need a few commands to do this.

> Prerequisite: You should have Git installed on your computer. 

---

## Keeping Your Branch Up to Date 

`main` → `my-branch`

To get the latest updates from `main` into your branch:

1. Switch to your branch:
   ```sh
   git checkout my-branch
   ```

2. Fetch updates from `main`:
   ```sh
   git fetch origin main
   ```

3. Merge them into your branch:
   ```sh
   git merge origin/main
   ```

If Git shows conflicts, open the listed files, make corrections, then run:
```sh 
git add .
git commit -m "Resolve merge conflicts"
```

---

## Pushing Changes from Your Branch to `main`

`my-branch` → `main`

Once your updates are tested, you may want to move selected changes into `main`.

1. Switch to `main`:
   ```sh
   # Make sure you're on your branch
   git checkout main
   
   # Save and stage your changes
   git add .
   git commit -m "Describe your changes"

   # Push your branch to GitHub
   git push origin my-branch
   ```

2. View your commit history:
   ```sh
   git log --oneline
   ```


---

## Summary of Key Commands

| Action                                   | Command                  |
|------------------------------------------|--------------------------|
| Switch to your branch                    | `git checkout my-branch` |
| Fetch updates from `main`                | `git fetch origin main`  |
| Merge changes into your branch           | `git merge origin/main`  |
| Pull updates from `main` (fetch + merge) | `git pull origin main`   |
| Switch to `main`                         | `git checkout main`      |
| View commit history                      | `git log --oneline`      |
| Push changes to `main`                   | `git push origin main`   |

---

## Final Tips

- Use `git status` to see what’s going on  
- Use `git log --oneline` to see commit history  
- Use Google or AI tools for troubleshooting  
- Ask teammates only after checking the basics

You don’t need to master Git to contribute — just keep your branch up to date and make your changes visible to the team in a clean and organized way.