# Installation

1. Clone the repository:
    - Use the terminal to go to the location you want to clone the repository.
    - Run the command: `git clone https://github.com/ESMAP-World-Bank-Group/EPM.git`


2. Create and activate branch:

We suggest to create a new branch for your work. This way, you can work on your own branch and merge it to the main branch when you are done.

    - To see the branch you are in, run the command: `git branch` 
    - Create (if it doesn't exist yet) and change to the branch you want to work on: `git checkout -b <branch_name>`
    - You’ve created a new branch locally and now want to push it to the remote repository: `git push -u origin <branch-name>``

3. Launch the model:

There are three ways to run the model:
- Using GAMS Studio (recommended for debugging)
- Using Excel (not recommended)
- Using Python (recommended for running the model)

There are two ways to include input data:
- A single .xlsx file with all data
- A folder with a list of .csv files with the data (recommended)
