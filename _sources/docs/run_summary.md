# Summary Command line

Use the following commands to install and run the EPM model for a new project.

| **Action** | **Command** |
|------------|------------|
| Clone the repository | `git clone https://github.com/ESMAP-World-Bank-Group/EPM.git` |
| Rename the `EPM` directory | `mv EPM EPM_Country` |
| Navigate to the `EPM_Country` directory | `cd EPM_Country` |
| Check the current branch | `git branch` |
| Create and switch to a new branch | `git checkout -b <branch-name>` |
| Activate Python Conda environment (when installed) | `conda activate esmap_env` |
| Navigate to the `epm` subdirectory | `cd epm` |
| Test the model using Python and default data input | `python epm.py --simple` |
| Fill with project-specific data input | Go to `input` and filled all `.csv` files |
| Debug the model in GAMS | Use GAMS Studio and run `main.gms` |
| Push to `origin/branch` to keep all modification. `-u` sets the upstream link, so future git push/pull commands work without specifying the branch. | `git push -u origin <branch-name>` |
| Run the model using Python with project-specific data input  | `python epm.py --folder_input data_xxx` |
| Check automatic results | Go to `output/simulation_xxxx/img` |
| Generate scenarios | Create `scenario.csv` in `input/data_xxx` for varying parameters |
| Run the model with scenarios | `python epm.py --folder_input data_xxx --scenarios` (uses `scenarios.csv`); provide a filename to override |
| Check scenario results | Go to `output/simulation_xxxx/img` |
| Check the result with Tableau | Go to Tableau Desktop and upload your simulation |
| Use Remote server to launch Sensitivity or Monte Carlo |  `python epm.py --sensitivity`|
