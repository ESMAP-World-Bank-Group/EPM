# Run EPM on the Remote Server

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

**Storage rules**

- Do **not** store data, code, or results in `/home/wb_yourID/`.
- Use the `/Data` directory for **all** storage and simulations. This is where disk space is allocated.

Hence, once on the server, navigate to the `/Data` folder to clone the repository. To do so, you should change working directory after connecting to the server:
```sh 
cd /Data
```
Then you can create your working directory and change your location in:
```sh
mkdir yourdirectory
```

Then you can clone the EPM repository:
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

The server is for running simulations, not for code development. Follow these steps to ensure a smooth workflow:

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

First time only: You need to add the shared Miniconda installation to your terminal environment. Run the following **once**:

To use `conda` from anywhere on this server, you need to add its location to your system `PATH`.
> This only needs to be done once. Just copy and paste the line below into your terminal.

```sh 
echo 'export PATH="/Data/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```
After running this, you can confirm conda is accessible:

```sh 
which conda
conda --version
```

All simulations must be run from the `/Data` folder, not from `/home`. 
To do so, you should change working directory after connecting to the server:

```sh 
cd /Data
```
Then you can create your working directory and change your location in:
```sh 
cd yourdirectory/EPM/epm
```

Each time you run:
```sh
conda activate epm_env
python epm.py
```

You can also specify arguments:
```sh
python epm.py --folder_input data_test --config input/data_test/config.csv --scenarios input/data_test/scenarios.csv --selected_scenario baseline
```

> **Important**: The AWS cluster currently only allows downloading files instead of directories. To ensure easier extraction of results, we recommend using the zip option to extract results folder.

To do so, use argument `--output_zip` when running the code. For instance:
```sh 
python epm.py --folder_input data_test --config input/data_test/config.csv --output_zip
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

For long-running simulations, it is important that your job continues running even if you disconnect from the server. This can be achieved using `tmux`, a terminal multiplexer that allows you to create virtual sessions that persist after logout.

Start a new tmux session, and then, inside the session, launch your job (adjust the command as needed):

```sh 
tmux new -s epmrun
python epm.py --folder_input data_test_region --config input/data_test_region/config.csv --sensitivity 
```


To leave the session without stopping your job, press the following key sequence:
```sh 
Ctrl + B, then D
```

This detaches the session and sends it to the background, allowing your job to continue running.

If the keyboard shortcut does not work (e.g., due to terminal configuration), you can also run the following from another terminal:
```sh 
tmux detach-client
```

To see all active tmux sessions:
```sh 
tmux list-sessions
```

If your session appears with `(attached)`, it is still active in a terminal window. If not, it is safely detached and running in the background.

To reconnect to a running session:
```sh 
tmux attach -t epmrun
```

To verify processes running:
```sh
ps aux | grep epm.py
```

You get a list of all active processes related to the script epm.py. Each line corresponds to a running process. Here's how to read one. Example line:
```sh 
wb636520  999873  6.9  0.0  891358 184220 ?  Sl  11:49  0:07 python epm.py 
```

Here is a column-by-column Breakdown

| Field | Column         | Description                                                                 |
|-------|----------------|-----------------------------------------------------------------------------|
|  wb636520     | `USER`      | User who launched the process                                               |
|   999873    | `PID`     | Process ID — unique identifier for the process                              |
|    6.9   | `%CPU`    | CPU usage percentage                                                        |
|  0.0     | `%MEM`    | Memory usage percentage                                                     |
|  891358     | `VSZ`     | Virtual memory size (in kilobytes)                                          |
|  184220     | `RSS`     | Resident Set Size — physical memory usage (in kilobytes)                   |
|   ?    | `TTY`     | Terminal controlling the process (`?` means none; typical for background)   |
|  Sl     | `STAT`    | Process status (e.g., `S` = sleeping, `R` = running, `Z` = zombie) + flags  |
|   11:49    | `START`   | Time the process started (HH:MM or date)                                    |
|   0:07    | `TIME`    | Total CPU time the process has used so far                                  |
|     python epm.py  | `COMMAND` | Command used to launch the process, including all arguments                 |


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
- **Delete directory and contents**: `rm -rf folder_name` *(⚠ irreversible)*

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
