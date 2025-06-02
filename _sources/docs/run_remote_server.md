# Running EPM on the Remote Server

The World Bank's remote server is designed for **running computationally heavy EPM simulations**.

## Server Specifications
- **CPU**: 4 cores  
- **Total RAM**: 31 GB  
- **Typical Free RAM**: ~6 GB  
- Use `free -h` and `top` to monitor real-time usage.

---

## 1. Prerequisites

You need:
- A **World Bank computer** or access via **VDI (Virtual Desktop Infrastructure)**
- A **YubiKey** (only for VDI login)

⚠️ If you're not using a WB-issued device, you **must** use the VDI and your YubiKey.

---

## 2. Connect to the Server

1. Go to [PrivX Login](https://privx.worldbank.org/auth/login)  
2. Sign in with **Microsoft credentials**
3. In the **Connections** tab, click on a host like `Linux-xxx-.worldbank.org`

Once connected, you’ll have access to:
- A **Terminal** (for commands)
- A **Files tab** (to upload/download files)

---

## 3. Clone the EPM Repository

Once on the server, navigate to your home directory and clone the repository:

```sh
cd ~
git clone https://github.com/ESMAP-World-Bank-Group/EPM.git
cd EPM
```

To clone a specific branch:
```sh
git clone --branch your-branch-name --single-branch https://github.com/ESMAP-World-Bank-Group/EPM.git
```

---

## 4. Best Practices Workflow

You do not modify the code on the sever directly. Instead, follow these steps to ensure a smooth workflow:

1. **Test Locally First**  
   Run a simple example on your computer before launching long runs on the server.

2. **Push Your Local Changes** from your computer
   ```sh
   git add .
   git commit -m "Your message"
   git push origin your-branch-name
   ```

3. **Update Code on the Server**
   ```sh
   cd ~/EPM
   git pull origin your-branch-name
   ```

---

## 5. Run EPM on the Server (Option 1 – Python or GAMS)

Once your code is ready, you can run EPM on the server using the **same steps as on your computer**.

### A. Python (Recommended)

First time only: create the Python environment

```sh
conda create -n esmap_env python=3.10
conda activate esmap_env
pip install -r requirements.txt
```

Each time you run:
```sh
conda activate esmap_env
cd EPM/epm
python run_epm.py
```

You can also specify arguments:
```sh
python epm.py --folder_input data_sapp --config input/data_sapp/config.csv --scenarios input/data_sapp/scenarios.csv --selected_scenario baseline
```

### B. GAMS (to test if bug does not appear in Python)
You don't have access to GAMS Studio on the server, but you can run GAMS directly from the terminal.
```sh
cd EPM/epm
gams main.gms
# or with arguments:
gams main.gms --FOLDER_INPUT input_folder
```

---

## 6. Run in Background (Essential for Long Runs)

To **start a long simulation and disconnect safely**, add `nohup` at the beginning and `&` at the end of your command:

```sh
nohup python epm.py --folder_input data_sapp --sensitivity &
```

This ensures the process keeps running even after you close the server session.

To verify it’s still running:
```sh
ps aux | grep epm.py
```

To stop it if needed:
```sh
kill -9 <PID>
```

---

## 7. Help Section: Terminal Commands

### File Navigation Basics

- **List files**: `ls`  
- **Detailed list**: `ls -l`  
- **Change directory**: `cd folder_name`  
- **Go up one level**: `cd ..`  
- **Print current directory**: `pwd`  
- **Make directory**: `mkdir new_folder`  
- **Delete directory and contents**: `rm -r folder_name` *(⚠ irreversible)*

### Server Usage Tips

- **Monitor usage**:  
  ```sh
  top        # real-time CPU/memory
  free -h    # memory summary
  ```
- **Find heavy processes**:  
  ```sh
  ps aux --sort=-%mem | head -10
  ```
- **Kill a process**:  
  ```sh
  kill -9 <PID>
  ```

### Final Reminders

✅ Test locally first  
✅ Always log out after use  
✅ Use `nohup` for long runs  
❌ Don’t overload the server  
🤝 Coordinate with others if needed
