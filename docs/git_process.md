# Git procedure

_Last update: Nov 2024_

## Creating a new branch
When running a new EPM version for a specific country or region, the recommended procedure is the following:
1. Create a new branch, with name matching the country or region. This new branch will match the latest version from the code available on the `main` branch.
2. Erase the `WB_EPM_v8_5_base.gms` and `WB_EPM_v8_5_Report.gms` files, which are specific to the country or region version. Simply copy paste your available files to the `country` branch instead.
3. Modify the `WB_EPM_input_readers.gms` file to match the specificities of the country version. 
   1. This may require some adaptation of input files. If using the CSV reader, need to break the available input excel files into corresponding csv files in the `input` folder.
   2. The modification of the `WB_EPM_input_readers.gms` file depends on whether you use the excel or the csv input files. If using the excel input files, the only part of the code that needs to be modified is when `READER = CONNECT_EXCEL`. If using the csv input files, two parts of the code need to be modified. First, when `READER = CONNECT_CSV`. Second, simply copy paste the changes you made to the section when `READER = CONNECT_CSV_PYTHON`. This will allow to use scenario specifications to launch simulations.

## Merging changing from the `main` branch to another branch
### Switch to your target branch:

First, make sure you're on the branch that needs to receive the changes:
```git checkout my-branch```

### Merge the changes from the other branch:

Fetch the specific branch: ``` git fetch origin other-branch ```

Then, merge the changes from the branch you want to pull from. Replace other-branch with the name of the branch you're pulling from.
```git merge origin/other-branch```


This may create some conflicts if you have changed your current branch. Resolve those conflicts to only include features you are interested in from the `main` branch.