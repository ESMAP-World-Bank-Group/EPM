# Git procedure

_Last update: Nov 2024_

## Running code with excel input file
### From python
When running the code from python, the function `launch_epm_multi_scenarios_excel` may be used. Here is an example of working code:
``` 
        PATH_GAMS = {
            'path_main_file': 'WB_EPM_v8_5_main.gms',  
            'path_base_file': 'WB_EPM_v8_5_base.gms',
            'path_report_file': 'WB_EPM_v8_5_Report.gms',
            'path_reader_file': 'WB_EPM_input_readers.gms',
            'path_excel_file': 'input/WB_EPM_8_5.xlsx',
            'path_cplex_file': 'cplex.opt'
        }
        launch_epm_multi_scenarios_excel()
```

### From GAMS
When running the code from GAMS Studio, the `main.gams` file must be called with the following arguments: 
```--READER CONNECT_EXCEL --BASE_FILE WB_EPM_v8_5_base.gms --XLS_INPUT input/WB_EPM_8_5.xlsx --REPORT_FILE WB_EPM_v8_5_Report.gms --READER_FILE WB_EPM_input_readers.gms```

The input file name may be modified depending on your input file.

## Creating a new branch
When running a new EPM version for a specific country or region, the recommended procedure is the following:
1. Create a new branch, with name matching the country or region. This new branch will match the latest version from the code available on the `main` branch.
2. Erase the `WB_EPM_v8_5_base.gms` and `WB_EPM_v8_5_Report.gms` files, which are specific to the country or region version. Simply copy paste your available files to the `country` branch instead.
3. Modify the `WB_EPM_input_readers.gms` file to match the specificities of the country version. 
   1. This may require some adaptation of input files. If using the CSV reader, need to break the available input excel files into corresponding csv files in the `input` folder.
   2. The modification of the `WB_EPM_input_readers.gms` file depends on whether you use the excel or the csv input files. If using the excel input files, the only part of the code that needs to be modified is when `READER = CONNECT_EXCEL`. If using the csv input files, two parts of the code need to be modified. First, when `READER = CONNECT_CSV`. Second, simply copy paste the changes you made to the section when `READER = CONNECT_CSV_PYTHON`. This will allow to use scenario specifications to launch simulations.

## Merge changes from the `main` branch to another branch
### Switch to your target branch:

First, make sure you're on the branch that needs to receive the changes:
```git checkout my-branch```

### Merge the changes from the other branch:

Fetch the specific branch: ``` git fetch origin other-branch ```

Then, merge the changes from the branch you want to pull from. Replace other-branch with the name of the branch you're pulling from.
```git merge origin/other-branch```


This may create some conflicts if you have changed your current branch. Resolve those conflicts to only include features you are interested in from the `main` branch.

## Push local changes to the `main` branch
1. Switch to the branch where you want to apply the changes (other-branch):  ```git checkout other-branch```
2. Cherry-pick the specific commits:
   - Use git log or a GUI tool to find the commit hash(es) containing the changes you want to transfer: ```git log```
   - Cherry-pick the specific commit(s) to `other-branch`: ```git cherry-pick <commit-hash>```
   - Resolve conflicts if necessary
   - Push changes to other-branch:
   - git push origin other-branch


