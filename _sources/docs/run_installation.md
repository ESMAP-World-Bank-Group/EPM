# EPM Installation Guide for Beginners  

This guide will walk you through installing and setting up the **EPM** model. It covers cloning the repository, creating your own branch, and launching the model.  

If you're new to Git, don't worry—just follow these steps carefully.  

---

## ✅ Summary of Key Commands  

| **Action** | **Command** |
|------------|------------|
| Clone the repository | `git clone https://github.com/ESMAP-World-Bank-Group/EPM.git` |
| Check the current branch | `git branch` |
| Create and switch to a new branch | `git checkout -b <branch-name>` |
| Push your new branch to remote | `git push -u origin <branch-name>` |
| Activate Conda environment | `conda activate esmap_env` |
| Run the model using Python | `python epm/run_epm.py` |
| Open and run the model in GAMS | Use GAMS Studio and run `main.gms` |

---

## 1️⃣ Clone the Repository  

To download the project to your local machine, follow these steps:  

1. Open a terminal or command prompt.  
2. Navigate to the directory where you want to store the project:  
   ```sh
   cd /path/to/your/folder
   ```
3. Clone the repository from GitHub:  
   ```sh
   git clone https://github.com/ESMAP-World-Bank-Group/EPM.git
   ```
4. Move into the newly created project directory:  
   ```sh
   cd EPM
   ```

---

## 2️⃣ Create and Activate a Branch  

A **branch** is like a separate workspace where you can make changes without affecting the main code (`main`).  

We recommend **creating your own branch** to work on, keeping your changes organized.  

### **Steps to Create and Activate Your Branch**  

1. **Check which branch you are currently on**:  
   ```sh
   git branch
   ```
   The active branch will have a `*` next to it.

2. **Create and switch to your own branch** (replace `<branch-name>` with a meaningful name, e.g., `guinea`):  
   ```sh
   git checkout -b <branch-name>
   ```
   This creates a new branch and switches to it.

3. **Push your branch to the remote repository** to save it online:  
   ```sh
   git push -u origin <branch-name>
   ```
   Now, your branch is stored remotely, and you can share your work if needed.

---

## 3️⃣ Launch the Model  

There are **three ways** to run the EPM model:  

### **1. Using GAMS Studio (Recommended for Debugging)**  
- Open GAMS Studio.  
- Navigate to the EPM folder.  
- Open and run `main.gms`.  

### **2. Using Python (Recommended for Multiple Scenarios)**  
1. **Activate the Conda environment** to ensure you have the required dependencies:  
   ```sh
   conda activate esmap_env
   ```
2. **Run the model using Python**:  
   ```sh
   python epm/run_epm.py
   ```
   Please check the data input structure in the `epm` folder before running the model.


---

## 🎯 Final Tips  

- **Use Git branches** to keep your work organized.  
- **Test your setup with a small dataset** before running large-scale simulations.  
- **Regularly fetch updates from `main`** to stay in sync with the latest changes.  
- **If you get stuck, check `git status` or `git log --oneline`** to understand what’s happening.  

By following these steps, you'll have EPM installed and running smoothly! 🚀  