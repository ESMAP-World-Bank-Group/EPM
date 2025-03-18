# 📌 Remote Server Specifications & Best Practices

**Be considerate of shared resources** 🤝  

## 🖥️ Server Specifications
Below are the key hardware specifications of the remote server (to update):

- **CPU Cores**: 4 cores
- **Total RAM**: `31.1 GB`: This is the total memory available for processes and system functions.
- **Free RAM**: `5.7 GB`: This is the portion of RAM that is completely unused at the moment.
- **Used RAM**: `2.3 GB`: This is the portion of RAM actively being used by running applications and the operating system kernel. 
- **Cached Memory**: `23.4 GB`: Cached Memory consists of files and programs that are kept in memory for faster access. Cached memory is technically available for use by other processes if needed.

To verify these details, you can run:
```sh
lscpu   # To check detailed CPU info
free -h # To check RAM usage in human-readable format
```

---

## 🛠️ Best Practices for Server Usage

### 📌 General Usage Guidelines:
1. **Be mindful of resource usage** 🏗️  
   - Check system load before running heavy tasks:  
     ```sh
     top
     htop  # If installed, provides a better view
     ```
   - Avoid long-running processes that consume excessive CPU or memory.

2. **Manage background processes properly** 🚀  
   - If running a script in the background, use:
     ```sh
     nohup python my_script.py > output.log 2>&1 &
     ```
   - List background jobs:
     ```sh
     jobs
     ps aux | grep my_script
     ```

3. **Release memory when possible** 🧹  
   - If a process is consuming too much RAM, consider terminating it:
     ```sh
     kill -9 <PID>
     ```
   - To find heavy processes:
     ```sh
     ps aux --sort=-%mem | head -10
     ```

4. **Be considerate of shared resources** 🤝  
   - Avoid monopolizing CPU and RAM.  
   - If multiple users are active, coordinate usage.

---

## 📝 Notes:
- To check system uptime and resource utilization:
  ```sh
  uptime
  top
  free -m
  ```

- If you experience high load, consult with the team before launching new processes.
"""

## Running a Command After Session Ends Using `nohup`

To run a command that continues running even after you log out, use `nohup`:

```bash
nohup your_command &
```

### Explanation:
- `nohup` → Prevents the process from being terminated when the session ends.
- `&` → Runs the command in the background.

### Example:
```bash
nohup python my_script.py &
```

### Checking the Running Process:
To confirm that the process is still running, use:
```bash
ps aux | grep my_script.py
```

### Viewing Logs (If Applicable):
If your script produces output and you didn't redirect it to a file, check the default `nohup.out`:
```bash
tail -f nohup.out
```

### Killing the Process (If Needed):
If you need to stop the running process, find its **PID** and kill it:
```bash
ps aux | grep my_script.py
kill -9 <PID>
```