
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
