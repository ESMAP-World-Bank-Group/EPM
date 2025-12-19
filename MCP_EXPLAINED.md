# MCP for EPM - Simple Explanation

## What is MCP?

**MCP (Model Context Protocol)** is a way for AI assistants (like the one in Cursor) to interact with your code and tools. Think of it as a bridge that lets the AI assistant run your Python scripts directly.

Instead of you having to:
1. Open terminal
2. Type: `python epm/epm.py --folder_input data_test --cpu 4`
3. Wait for results
4. Copy/paste results back to Cursor

With MCP, you can just **ask the AI in Cursor** to run EPM, and it will do it for you!

## What I Created

I created two things:

### 1. `mcp_server.py` - The Bridge Script
This is a Python script that:
- Listens for commands from Cursor's AI
- Translates those commands into actual `epm.py` calls
- Runs `epm.py` with the right parameters
- Returns the results back to Cursor

### 2. Configuration Instructions
Instructions on how to tell Cursor where to find this bridge script.

## How It Works (Simple Version)

```
You (in Cursor) 
    ↓
    "Run EPM with data_test folder and 4 CPUs"
    ↓
Cursor's AI
    ↓
MCP Server (mcp_server.py)
    ↓
Executes: python epm/epm.py --folder_input data_test --cpu 4
    ↓
Results come back to you in Cursor
```

## Step-by-Step Setup

### Step 1: Install the MCP Package

Open your terminal and run:
```bash
cd /Users/lucas/Documents/World\ Bank/Projects/EPM_APPLIED/EPM_main
pip install mcp
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Step 2: Find Cursor's Configuration File

The configuration file location depends on your system:

**macOS:**
```
~/Library/Application Support/Cursor/User/globalStorage/mcp.json
```

**Windows:**
```
%APPDATA%\Cursor\User\globalStorage\mcp.json
```

**Linux:**
```
~/.config/Cursor/User/globalStorage/mcp.json
```

**To find it quickly:**
1. Open Cursor
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Open User Settings"
4. Look for the settings folder path

### Step 3: Create/Edit the MCP Configuration

1. Navigate to the folder mentioned above
2. Create a file called `mcp.json` (if it doesn't exist)
3. Add this content (update the path to match YOUR project path):

```json
{
  "mcpServers": {
    "epm": {
      "command": "python",
      "args": [
        "/Users/lucas/Documents/World Bank/Projects/EPM_APPLIED/EPM_main/mcp_server.py"
      ],
      "cwd": "/Users/lucas/Documents/World Bank/Projects/EPM_APPLIED/EPM_main",
      "env": {}
    }
  }
}
```

**Important:** 
- Replace the path with YOUR actual project path
- Make sure to use forward slashes `/` or escape spaces properly
- The `cwd` tells Python where to run from (so relative imports work)

### Step 4: Restart Cursor

Close and reopen Cursor completely for the changes to take effect.

## How to Use It

Once set up, you can use it in two ways:

### Method 1: Ask the AI Directly

Just type in Cursor's chat:
```
"Run EPM with the data_test folder using 2 CPUs"
```

or

```
"Run EPM with:
- folder_input: data_test
- scenarios: scenarios.csv  
- cpu: 4
- sensitivity: true"
```

### Method 2: Use the Tools Directly

The AI will have access to two "tools":
1. **`run_epm`** - Full control with all options
2. **`run_epm_simple`** - Quick test with defaults

## Example Conversations

### Example 1: Simple Run
**You:** "Run EPM with the default test data"

**AI:** *Uses the tool to run:*
```bash
python epm/epm.py --folder_input data_test --cpu 1
```

### Example 2: Complex Run
**You:** "Run EPM with data_test_region, enable sensitivity analysis, use 4 CPUs, and run only the baseline scenario"

**AI:** *Uses the tool to run:*
```bash
python epm/epm.py --folder_input data_test_region --sensitivity --cpu 4 --selected_scenarios baseline
```

## Testing if It Works

### Test 1: Check if MCP server runs
```bash
cd /Users/lucas/Documents/World\ Bank/Projects/EPM_APPLIED/EPM_main
python mcp_server.py
```

If it starts without errors and waits (doesn't exit), it's working!

### Test 2: Ask Cursor AI
After restarting Cursor, try asking:
```
"Can you run EPM with the data_test folder?"
```

If the AI responds and actually runs the command, it's working!

## Troubleshooting

### Problem: "MCP server not found"
**Solution:** 
- Check the path in `mcp.json` is correct
- Make sure Python is in your PATH
- Try using absolute paths

### Problem: "Import error: mcp"
**Solution:**
```bash
pip install mcp
```

### Problem: "AI doesn't recognize the tool"
**Solution:**
- Make sure you restarted Cursor completely
- Check the `mcp.json` file syntax is valid JSON
- Look for errors in Cursor's developer console (Help → Toggle Developer Tools)

### Problem: "EPM runs but fails"
**Solution:**
- This is normal - the MCP server just runs EPM, it doesn't fix EPM errors
- Check that GAMS is installed
- Verify your input data files exist
- Check the error messages returned by the tool

## What You Can Do With It

Once working, you can:
- ✅ Run EPM directly from Cursor chat
- ✅ Ask the AI to run multiple scenarios
- ✅ Get results back in the chat
- ✅ Have the AI analyze the results
- ✅ Chain operations: "Run EPM, then analyze the output"

## Alternative: Simple Test Script

If MCP seems too complex, I can also create a simple test script that you can run directly. Would you like that instead?

