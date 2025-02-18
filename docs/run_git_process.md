# Git procedure

_Last update: Nov 2024_

## Merge changes from the `main` branch to my branch (`my-branch`): `main` --> `my-branch`

I want to update `my-branch` with new interesting features developed in the `main` branch.

### Switch to your target branch:

First, make sure you're on the branch that needs to receive the changes:
```git checkout my-branch```

### Fetch the changes from the other branch:

Second,  _Get the latest changes from the remote repository, but don’t apply them to my current branch yet, what is called Fetch._ 

Fetch the specific branch: ``` git fetch origin main```

This command:
- downloads commits, files, and references from the remote repository.
- it updates your local copy of the remote branch but does not merge or modify your working branch.

### Merge the changes from the other branch:

Then, merge the changes from the branch you want to pull from.
```git merge origin/main```

This may create some conflicts if you have changed your current branch. Resolve those conflicts to only include features you are interested in from the `main` branch.

## Push local changes from my branch (`my-branch`) to the `main` branch: `my-branch` --> `main`

I want to release my new development on the `main' branch, so that others (and myself in the future) can use it.

1. Switch to the branch where you want to apply the changes, here `main`:  ```git checkout main```
2. Cherry-pick the specific commits:
   - Use git log or a GUI tool to find the commit hash(es) containing the changes you want to transfer: ```git log```
   - Cherry-pick the specific commit(s) to `main`: ```git cherry-pick <commit-hash>```
   - Resolve conflicts if necessary
   - Push changes to other-branch: ```git push origin main```


