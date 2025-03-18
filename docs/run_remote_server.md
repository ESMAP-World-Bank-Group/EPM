# Running EPM with Remote Server

Our calculation server is the preferred way to launch computationally expensive EPM simulations.

## Prerequisites
Before you connect to the remote server, ensure you have the following:
- A **WB computer** or **VDI (Virtual Desktop Infrastructure)**.
- A **YubiKey** (only required for VDI authentication).

## Steps to Connect

1. **Access the PrivX Portal**
   - Open your browser and navigate to:  
     **[PrivX Login](https://privx.worldbank.org/auth/login)**
   - Click on the **Microsoft Sign-in** option and log in with your credentials.

2. **Connect to the Server**
   - After logging in, go to the **Connections** tab.
   - Click on the **host server** that starts with `"Linux-xxx-.worldbank.org"`.

3. **Accessing the Server**
   - You are now connected to the remote server.
   - The interface provides:
     - A **terminal** for executing commands.
     - A **Files** section (accessible via the top-right header) to manage files.

## Navigating the File Interface
The **Files** tab in the remote connection interface allows you to:
- **Browse directories** by clicking on folders.
- **Upload files** from your local computer to the server.
- **Download files** from the server to your local machine.
- **Edit files** directly through the interface if needed.

However, for more control, you can manage files directly from the terminal.

## Basic Terminal Commands

### Listing Files and Navigating Directories
- **List files and folders**:  
  ```sh
  ls
  ```
- **Show detailed list (permissions, owner, size, modification date)**:  
  ```sh
  ls -l
  ```
- **Change directory**:  
  ```sh
  cd folder_name
  ```
- **Move up one directory level**:  
  ```sh
  cd ..
  ```
- **Show the current directory path**:  
  ```sh
  pwd
  ```

### Creating and Removing Directories
- **Create a new directory**:  
  ```sh
  mkdir new_directory
  ```
- **Remove an empty directory**:  
  ```sh
  rmdir directory_name
  ```
- **Remove a directory and its contents**:  
  ```sh
  rm -r directory_name
  ```
  ⚠️ **Be careful with `rm -r`, as it will permanently delete all files inside the directory!**

## Uploading and Downloading Files
To transfer files between your local machine and the remote server, you can use:
- **The web interface Files tab** (simpler for occasional uploads/downloads).

## Cloning the EPM Repository
To work with EPM, first clone the repository from GitHub:

```sh
git clone https://github.com/ESMAP-World-Bank-Group/EPM.git
```

This will create a directory named `EPM` with all the necessary files.

## Best Practices for Running EPM
When working with EPM, follow this workflow:

1. **Test Locally First**  
   - Run your code on a simple example (e.g., a few years, LP, one scenario) to ensure it works correctly before using the server.
   - Keep the test case small to debug efficiently.

2. **Push Changes to Your Git Branch**  
   - Once your local test runs successfully, push your changes to your remote branch:
     ```sh
     git add .
     git commit -m "Your commit message"
     git push origin your-branch-name
     ```

3. **Pull Your Updated Code on the Server**  
   - Navigate to your EPM directory on the server:
     ```sh
     cd EPM
     ```
   - Update the repository:
     ```sh
     git pull origin your-branch-name
     ```
   This ensures you have the latest version of your code on the server.


4. **Run EPM on the Server**  
   - Once your updated code is on the server, you can launch your EPM simulations.

## Running EPM

There are two ways to launch EPM:

### 1. Running EPM Directly with GAMS
This method allows you to execute the model directly using GAMS.

- Navigate to the EPM directory:
  ```sh
  cd EPM/epm
  ```
- Run the GAMS model:
  ```sh
  gams main.gms
  ```
- You can also provide inline arguments:
  ```sh
  gams main.gms --FOLDER_INPUT data
  ```

### 2. Running EPM Using the Python API (Recommended)
Using the Python API enables additional functionalities such as direct scenario analysis and sensitivity analysis. You don’t need to know Python—just follow these steps !
Refer to EPM Python API documentation for more details.

- Navigate to the EPM directory:
  ```sh
  cd EPM
  ```
- Activate the Conda environment to ensure all necessary Python packages are available:
  ```sh
  conda activate esmap_env
  ```
- Run the EPM model:
  ```sh
  python epm/run_epm.py
  ```
- You can also specify additional arguments when launching EPM.


---
**Reminder**: Always test your code locally before running large simulations on the server to avoid unnecessary computational load.

---
**Troubleshooting:**
- If you cannot log in via **Microsoft Sign-in**, ensure your credentials are correct.
- If the VDI option is used, make sure your **YubiKey** is properly inserted and functioning.
- If you cannot access the terminal, try reconnecting to the server.

---
**Reminder**: Always log out when you're finished to ensure security.