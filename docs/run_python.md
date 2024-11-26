
# Run EPM

1 Clone the repository:
    - Use the terminal to go to the location you want to clone the repository.
    - Run the command: `git clone https://github.com/ESMAP-World-Bank-Group/EPM.git`

2. Change branch:
    - To see the branch you are in, run the command: `git branch` 
    - Create (if it doesn't exist yet) and change to the branch you want to work on: `git checkout -b <branch_name>`
    - You’ve created a new branch locally and now want to push it to the remote repository: `git push -u origin <branch-name>``



2. Use Python:
    - Activate environement using `requirements.txt` file:
        - Run the command: `pip install -r requirements.txt`
    
3. Include input
    - Depending on your way to run the model, you can include:
        - A single .xlsx file with all data
        - A folder with a list of .csv files with the data (recommended for sensitivity analysis)


Tips using git:
