# EPM Installation Guide

This guide walks you through installing and setting up the **EPM** model. It covers:

- Cloning the repository (getting the code on your computer)
- Creating your own branch (to work independently)
- Launching the model (via GAMS or Python)

> You don’t need to be familiar with Git—just follow the steps carefully.

---

## Quick Install (Windows Beta)

If you are on **Windows** you can try the one-click installer first:

1. [Download `setup_epm.bat`](dwld/setup_epm.bat) and place it in an empty folder.
2. Double-click the file (or right-click → **Run as administrator**) to launch the guided setup.
3. Wait for the script to finish—it checks for Git, Conda, and GAMS, pulls the latest EPM code, recreates the `epm_env` Conda environment, installs Python dependencies, and runs a quick GAMS/Python smoke test.
4. When the green success message appears, open the cloned `EPM` folder and continue with the rest of this guide.

> The installer writes a log file next to the script (`setup_log.txt`). If it stops with an error, or if you are on another operating system, follow the manual steps below instead.

## Quick Install (macOS Beta)

If you are on **macOS** you can use the shell installer:

1. [Download `setup_epm.sh`](dwld/setup_epm.sh) and place it in an empty folder (e.g., `~/EPM_Setup`).
2. Open **Terminal**, change into that folder (`cd ~/EPM_Setup`), and make the script executable:
   ```sh
   chmod +x setup_epm.sh
   ```
3. Run the installer:
   ```sh
   ./setup_epm.sh
   ```
4. The script verifies Git, Conda, and GAMS, pulls the latest EPM repository, rebuilds the `epm_env` Conda environment, installs Python dependencies, and runs a quick GAMS/Python smoke test. When it reports success, open the cloned `EPM` folder to continue with the manual steps below.

> The macOS installer also writes `setup_log.txt` alongside the script. If the run fails or you prefer to configure things manually, proceed with the next section.

---

## 1. Clone the Repository

To get the project on your computer:

1. Open **Terminal** (macOS) or **Command Prompt** (Windows).

   - On **macOS**: `Applications > Utilities > Terminal`
   - On **Windows**: Search "Command Prompt" in the Start menu

2. Navigate to the folder where you want to install the project.  
   This tells your computer where to download the files.

   Example:

   ```sh
   cd path/to/your/Projects
   ```

   Or step-by-step:

   ```sh
   cd Documents
   cd Projects
   ```

3. Clone the repository:

   ```sh
   git clone https://github.com/ESMAP-World-Bank-Group/EPM.git
   ```

4. Move into the project folder:
   ```sh
   cd EPM
   ```

---

## 2. Create and Activate a Branch

A **branch** is your personal workspace, so you can make changes without affecting the main version of the code.

1. Check your current branch:

   ```sh
   git branch
   ```

2. Create and switch to your new branch (e.g., `guinea_2025`):
   ```sh
   git checkout -b guinea_2025
   ```

If you check your current branch again, you should now see the name of your new branch highlighted.

3. Push your new branch to GitHub:
   ```sh
   git push -u origin guinea_2025
   ```

You can check the branches on GitHub to confirm your new branch is there.
This command sets the upstream branch for future pushes, so you can use `git push` without specifying the branch name next time.

```sh
git push
```

---

## 3. Test the Installation

The project includes a test dataset (`data_test`) for Guinea to verify everything works.

### Option 1 – Using GAMS Studio (Recommended for First-Time Users)

1. Open **GAMS Studio**
2. Use the file browser to open the downloaded `EPM` folder
3. Open `main.gms`
4. Click **Run**

### Option 2 – Using Python (For Advanced Users)

The Python API enables advanced automation and custom analyses.

See the next section: [Running EPM from Python](#) for more details.

---

Next steps are described with high-level instructions, but you can find detailed steps in the documentation.

## 4. Run EPM with Your Own Data

Once the test case works, you can run EPM with your own input data.

- Input files should be placed in a new folder inside the `input/` directory
- A good starting point is to **duplicate an existing folder** (e.g., `data_test`) and modify its content
- Detailed instructions are available in the **Input section** of the documentation

EPM can be run with your data either via **GAMS Studio** or **Python** as previously described.

---

## 5. Advanced Usage with Python

The Python API supports advanced features such as:

- Scenario generation
- Sensitivity analysis
- Monte Carlo simulations

Refer to the **Run EPM from Python** section for usage examples and command-line options.

---

## 6. Troubleshooting

If you encounter issues:

- Check the **Troubleshooting** section in the documentation
- Use **AI tools or Google** to look up the error message
- If you're still stuck, reach out to the EPM team

To debug input issues, you can also use GAMS Studio’s trace mode. (See the debugging tips section.)

---

## 7. Analyzing Results

After running the model:

- Results are automatically saved in the `output/` folder
- Use the summary `.csv` files for key outputs
- For more detailed analysis, you can connect the CSVs to **Tableau** or another visualization tool

---

## 8. Run on the Remote Server

Once your setup works correctly and has been tested on a small test case, you can run large-scale simulations on the **remote server**.

This is especially useful for:

- Running long or heavy simulations (e.g., full-year, multi-zone, Monte Carlo)
- Avoiding performance limitations on your local machine

Refer to the **Remote Server Usage** section of the documentation for:

- Connection instructions
- Best practices
- How to launch the model using GAMS or Python on the server

Make sure your local run works before using the server, to avoid unnecessary load and easier debugging.

---

## 9. Contribute Improvements to the Main Branch

If you've developed new features or made improvements that should be shared:

- Follow the Git contribution workflow (e.g., pull request from your branch)
- Make sure your code is tested and documented
- If unsure, contact the EPM team for help with integration

Our goal is to make the framework collaborative and maintainable. Contributions are welcome!
