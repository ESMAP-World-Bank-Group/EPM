# Git procedure

_Last update: Nov 2024_

## Creating a new branch
When running a new EPM version for a specific country or region, the recommended procedure is the following:
1. Create a new branch, with name matching the country or region. This new branch will match the latest version from the code available on the `main` branch.
2. Erase the `WB_EPM_v8_5_base.gms` and `WB_EPM_v8_5_Report.gms` files, which are specific to the country or region version. Simply copy paste your available files to the `country` branch instead.
3. Modify the `WB_EPM_input_readers.gms` file to match the specificities of the country version. 
   1. This may require some adaptation of input files. If using the CSV reader, need to break the available input excel files into corresponding csv files in the `input` folder.
   2. The modification of the `WB_EPM_input_readers.gms` file depends on whether you use the excel or the csv input files. If using the excel input files, the only part of the code that needs to be modified is when `READER = CONNECT_EXCEL`. If using the csv input files, two parts of the code need to be modified. First, when `READER = CONNECT_CSV`. Second, simply copy paste the changes you made to the section when `READER = CONNECT_CSV_PYTHON`. This will allow to use scenario specifications to launch simulations.

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


