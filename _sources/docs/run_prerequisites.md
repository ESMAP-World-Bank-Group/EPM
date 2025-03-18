
# Prerequisites: Install Required Tools  

Before starting, ensure you have the following installed on your computer:  

### 1️⃣ **Git (Version Control System)**  
Git is used to download and manage versions of the EPM repository.  

- **Windows**: [Download Git for Windows](https://git-scm.com/download/win)  
- **Mac**: Install using Homebrew:  
  ```sh
  brew install git
  ```
- **Linux**: Install using your package manager:  
  ```sh
  sudo apt install git  # Ubuntu/Debian
  sudo dnf install git  # Fedora
  ```
- **Check if Git is installed**:  
  ```sh
  git --version
  ```
  If installed, this will display the Git version.

### 2️⃣ **Python & Conda (Required for Running the Model in Python)**  
Python is needed to execute the model using scripts. If you plan to only use GAMS, you can skip this step. 

- **Install Miniconda (Recommended)**:  
  [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it for your operating system.  

- **Verify installation**:  
  ```sh
  python --version
  conda --version
  ```

### 3️⃣ **GAMS (If Using GAMS Studio for Debugging)**  
- **Download & install GAMS**: [GAMS Website](https://www.gams.com/download/)  

#### Adding GAMS to the System PATH

GAMS (General Algebraic Modeling System) needs to be added to the system PATH to be accessible from the command line. Below are the steps to add GAMS to the PATH for different operating systems.

##### Windows

1. **Locate the GAMS Installation Directory:**
   - By default, GAMS is installed in `C:\\GAMS\\<version>`. 
   - If unsure, open the GAMS IDE and check the installation path in `Help -> About`.

2. **Open Environment Variables:**
   - Press `Win + R`, type `sysdm.cpl`, and press `Enter`.
   - Go to the `Advanced` tab and click on `Environment Variables`.

3. **Edit the System PATH Variable:**
   - Under `System Variables`, find the variable named `Path` and click `Edit`.
   - Click `New` and add the full path to the GAMS installation directory (e.g., `C:\\GAMS\\40.1`).

4. **Apply and Test:**
   - Click `OK` to save the changes and close all dialog boxes.
   - Open `Command Prompt` and type:
     ```sh
     gams
     ```
   - If installed correctly, GAMS should run.

##### macOS

1. **Locate the GAMS Installation Directory:**
   - Typically, GAMS is installed in `/Applications/GAMS<version>`. 

2. **Edit the Shell Configuration File:**
   - Open Terminal and edit the shell profile:
     ```sh
     nano ~/.zshrc  # For Zsh (default in macOS Catalina and later)
     nano ~/.bash_profile  # For Bash (older macOS versions)
     ```

3. **Add GAMS to the PATH:**
   - Add the following line to the file:
     ```sh
     export PATH="/Applications/GAMS<version>:$PATH"
     ```
   - Replace `<version>` with the correct installed version.

4. **Apply and Test:**
   - Save and exit (`CTRL + X`, then `Y`, then `Enter`).
   - Run:
     ```sh
     source ~/.zshrc  # or source ~/.bash_profile
     gams
     ```
   - If installed correctly, GAMS should run.
