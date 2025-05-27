# EPM Installation Guide

This guide walks you through installing and setting up the **EPM** model. It covers:
- Cloning the repository (getting the code on your computer)
- Creating your own branch (to work independently)
- Launching the model (from GAMS or Python)

You don’t need to be familiar with Git—just follow the steps carefully.

---

## 1. Clone the Repository

To get the project on your computer:

1. Open **Terminal** (on macOS) or **Command Prompt** (on Windows). All following commands should be run in this terminal window.

   - On **macOS**, you can find Terminal in `Applications > Utilities > Terminal`.
   - On **Windows**, search for "Command Prompt" in the Start menu.

2. Choose where you want to download the EPM project.

   > This step is called "navigating"—it simply tells your computer where to install the files.  
   > For example, if you want to install it in a folder called `Projects`, type:
   ```sh
   cd path/to/your/Projects
   ```
   Replace `WB/Projects` with the actual folder path. You can create the folder first using your file explorer if needed.
   You can also use `cd` iteratively to navigate through folders, like:
   ```sh
   cd WB
   cd Projects
   ```

3. Download the code by typing:
   ```sh
   git clone https://github.com/ESMAP-World-Bank-Group/EPM.git
   ```
   You need to have **Git** installed on your computer for this step (see Pre-requisites).
   This step takes the latest version of the EPM code from [GitHub](https://github.com/ESMAP-World-Bank-Group/EPM.git) and puts it in a folder called `EPM` in your chosen directory.

4. Move into the EPM folder:
   ```sh
   cd EPM
   ```

---

## 2. Create and Activate a Branch

A **branch** is your personal workspace. It lets you make changes without affecting the main project.

### Steps:

1. Check which branch you’re on:
   ```sh
   git branch
   ```
   The current branch will have a `*` next to it.

2. Create and switch to a new branch (replace `<branch-name>` with something like `guinea`):
   ```sh
   git checkout -b <branch-name>
   ```

3. Save your branch online:
   ```sh
   git push -u origin <branch-name>
   ```

---

## 3. Launch the Model

The project includes sample data so you can check that everything is working.

If this is your first time using EPM, we suggest starting with **GAMS Studio** to make sure everything runs properly.

### Option 1: Using GAMS Studio (Recommended for First-Time Users)

- Open **GAMS Studio**.
- Use the file explorer in GAMS to open the `EPM` folder you downloaded.
- Open the file `main.gms`.
- Click **Run**.

### Option 2: Using Python (Recommended for Advanced Users)

Python allows more flexibility and automation.

→ See the next section: **Running EPM from Python**.