# Run on Remote Server

The World Bank remote server is designed for computationally heavy simulations: large scenarios, Monte Carlo runs, or anything that would be too slow on a laptop.

## Prerequisites

- A World Bank computer, or VDI access with a YubiKey
- EPM tested and working locally before running on the server

---

## 1. Connect

1. Go to [privx.worldbank.org/auth/login](https://privx.worldbank.org/auth/login)
2. Sign in with your Microsoft credentials
3. In the **Connections** tab, select a host (e.g. `Linux-xxx-.worldbank.org`)

Once connected you have access to a **Terminal** and a **Files** tab for uploads/downloads.

---

## 2. Clone the repository

All data and code must be stored in `/Data`, not in `/home`.

```sh
cd /Data
mkdir your_project_folder
cd your_project_folder
git clone --branch your-branch-name --single-branch https://github.com/ESMAP-World-Bank-Group/EPM.git
cd EPM/epm
```

---

## 3. Keep code in sync

The server is for running simulations, not for development. Work locally, then sync:

```sh
# On your local machine
git add .
git commit -m "ready for server run"
git push origin your-branch-name

# On the server
cd /Data/your_project_folder/EPM
git pull origin your-branch-name
```

---

## 4. Run EPM

!!! warning "RAM usage"
    The cluster has 96 GB of total RAM. Individual runs can consume 10+ GB each. Monitor usage carefully when running in parallel or alongside other users.

### Python (recommended)

First time only, add Miniconda to your PATH:

```sh
echo 'export PATH="/Data/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Then run:

```sh
conda activate epm_env
cd /Data/your_project_folder/EPM/epm
python epm.py --folder_input your_data --config your_data/config.csv --scenarios --cpu 8 --output_zip
```

Use `--output_zip` to compress results into a single file (the server only allows downloading files, not folders).

### GAMS (for low-level debugging only)

```sh
cd /Data/your_project_folder/EPM/epm
gams main.gms --FOLDER_INPUT your_data
```

---

## 5. Run in background

For long runs, use `tmux` so the job continues if you disconnect:

```sh
tmux new -s epmrun
# launch your command here
python epm.py --folder_input your_data --config your_data/config.csv --output_zip
```

Detach without stopping the job: `Ctrl + B`, then `D`

```sh
tmux attach -t epmrun      # reconnect to the session
tmux list-sessions          # see all active sessions
```

To check if your job is still running:

```sh
ps aux | grep epm.py
```

Find your username in the output; if the line is there, the job is running. To stop it:

```sh
kill -9 <PID>
```

---

??? note "Terminal command reminder"

    | Action | Command |
    |---|---|
    | List files | `ls` or `ls -l` |
    | Change directory | `cd folder_name` |
    | Go up one level | `cd ..` |
    | Print current path | `pwd` |
    | Create folder | `mkdir folder_name` |
    | Delete folder | `rm -rf folder_name` |
    | Monitor CPU/RAM | `top` or `free -h` |
    | Find heavy processes | `ps aux --sort=-%mem \| head -10` |
