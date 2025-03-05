# Git Procedure for Beginners

## Introduction

Git is a **version control system** that helps developers manage changes to their code. One of the key features of Git is **branches**.

### What is a Branch?
A **branch** is like a separate workspace where you can make changes to the code **without affecting the main version**. This allows multiple people to work on different features simultaneously.

There are two main types of branches:
- **Local branch**: Exists only on your computer.
- **Remote branch**: A shared version of the branch stored on GitHub or another Git server.

When working with Git, you often need to **merge updates from the `main` branch** into your branch or **push your own changes** to `main`. These steps ensure your code stays up-to-date and organized.

🚀 **Good news:** You only need a few Git commands to do this. Just follow these steps!

---

## Prerequisite: Install Git

Before using Git, you need to install it on your computer:

- **Windows**: [Download Git for Windows](https://git-scm.com/download/win)
- **Mac**: Install Git using Homebrew:
  ```sh
  brew install git
  ```
- **Linux**: Install Git using your package manager:
  ```sh
  sudo apt install git  # Ubuntu/Debian
  sudo dnf install git  # Fedora
  ```

To check if Git is installed, run:
```sh
git --version
```
If installed, this will show the Git version.

---

## Merging Updates from the `main` Branch to Your Branch (`my-branch`)

When working with Git, other team members might add new features or fixes to the main branch. If your branch (my-branch) is based on an older version of the project, it won’t include these updates.

To keep your branch up to date, you need to merge changes from main into my-branch. This ensures that you are working with the latest code and reduces the chances of conflicts when you later push your own changes to main.

**Flow: `main` → `my-branch`**  
👉 You want to update `my-branch` with the latest changes from the `main` branch.

### 1️⃣ Switch to Your Branch
Before merging updates, make sure you’re on `my-branch`:
```sh
git checkout my-branch
```

### 2️⃣ Fetch Updates from the Remote Repository
Before merging, **fetch** the latest changes from the remote repository without applying them yet:
```sh
git fetch origin main
```
🔹 This command:
- Downloads new commits from `main` **without modifying your code**.
- Lets you see the latest changes before merging.

### 3️⃣ Merge Changes from `main`
Now, apply the updates from `main` to your branch:
```sh
git merge origin/main
```
💡 **What happens here?**
- Git tries to automatically combine the changes from `main` into `my-branch`.
- If there are **no conflicts**, the merge happens smoothly.
- If there are **conflicts** (because both branches changed the same part of a file), you must **manually resolve them**.

✅ **Tip for Beginners**: If a merge conflict happens, Git will show the conflicting files. Open them in your code editor, make the necessary changes, then **commit** the resolved files.

By following these steps regularly, you ensure that your code remains synchronized with the latest development in main. 🚀

---

## Pushing Your Changes to the `main` Branch

After making changes in your branch (my-branch), you might want to publish them to the main branch so that others can use your updates.

This process ensures that your new features, bug fixes, or improvements become part of the main codebase. However, instead of merging everything at once, it’s often best to selectively apply specific changes using cherry-picking. This allows you to pick only the relevant commits that should be included in main, avoiding unnecessary modifications.

**Flow: `my-branch` → `main`**  
👉 Now, you want to **publish** your new feature from `my-branch` to `main` so others can use it.

### 1️⃣ Switch to the `main` Branch
First, move to the `main` branch:
```sh
git checkout main
```

### 2️⃣ Cherry-Pick Specific Commits
Instead of merging everything, you may want to **selectively pick specific commits** from `my-branch`.

1. View your commit history:
   ```sh
   git log --oneline
   ```
   This will display a list of commits with unique **commit hashes** (e.g., `a1b2c3d4`).

2. Pick and apply specific commits:
   ```sh
   git cherry-pick <commit-hash>
   ```
   Replace `<commit-hash>` with the actual commit ID you want to apply.

3. If there are conflicts, resolve them as before.

### 3️⃣ Push Changes to `main`
After cherry-picking the changes, push them to the remote repository:
```sh
git push origin main
```
Now, your new features are part of `main`, and everyone can use them! 🎉

---

## Summary of Key Commands

| Action                        | Command |
|--------------------------------|-------------------------------------------|
| Switch to your branch          | `git checkout my-branch` |
| Fetch latest updates from `main` | `git fetch origin main` |
| Merge changes from `main`      | `git merge origin/main` |
| Switch to `main`               | `git checkout main` |
| View commit history            | `git log --oneline` |
| Cherry-pick a specific commit  | `git cherry-pick <commit-hash>` |
| Push changes to `main`         | `git push origin main` |

---

## Final Thoughts

Git can seem complex, but you **only need a few essential commands** to work effectively. Just **follow the steps above**, and you'll be able to:
✅ Keep your branch up to date  
✅ Merge changes from `main`  
✅ Selectively add your work to `main`  

If you get stuck, remember:  
- **`git status`** → Shows what’s happening  
- **`git log --oneline`** → Shows your commit history  
- **Google & GitHub Docs** → Are your best friends  

Happy coding! 🚀

