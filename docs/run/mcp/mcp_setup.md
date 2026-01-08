# MCP Server Setup for EPM

This guide explains how to set up and use the MCP (Model Context Protocol) server to run `epm.py` directly from Cursor.

## Installation

1. **Install the MCP package:**
   ```bash
   pip install mcp
   ```
   
   Or install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Make the MCP server executable (optional):**
   ```bash
   chmod +x tools/mcp_server.py
   ```

## Configuration in Cursor

To use the MCP server in Cursor, you need to add it to Cursor's MCP configuration. The configuration file location depends on your system:

### macOS
`~/Library/Application Support/Cursor/User/globalStorage/mcp.json`

### Windows
`%APPDATA%\Cursor\User\globalStorage\mcp.json`

### Linux
`~/.config/Cursor/User/globalStorage/mcp.json`

### Configuration Content

Add the following to your MCP configuration file:

```json
{
  "mcpServers": {
    "epm": {
      "command": "python",
      "args": [
        "/Users/lucas/Documents/World Bank/Projects/EPM_APPLIED/EPM_main/tools/mcp_server.py"
      ],
      "env": {}
    }
  }
}
```

**Important:** Update the path in `args` to match your actual project path.

Alternatively, you can use a relative path if you set the `cwd`:

```json
{
  "mcpServers": {
    "epm": {
      "command": "python",
      "args": ["tools/mcp_server.py"],
      "cwd": "/Users/lucas/Documents/World Bank/Projects/EPM_APPLIED/EPM_main",
      "env": {}
    }
  }
}
```

## Usage

Once configured, you can use the MCP server in Cursor to:

1. **Run EPM with full configuration:**
   - Use the `run_epm` tool with various parameters like:
     - `folder_input`: Input data folder
     - `config`: Configuration file
     - `scenarios`: Scenario file
     - `cpu`: Number of CPU cores
     - `sensitivity`: Enable sensitivity analysis
     - `montecarlo`: Enable Monte Carlo analysis
     - And more...

2. **Run EPM with simple/default settings:**
   - Use the `run_epm_simple` tool for quick tests with minimal configuration

## Example Usage

### Simple Run
```
Run EPM with folder_input="data_test" and cpu=2
```

### Full Configuration Run
```
Run EPM with:
- folder_input: "data_test_region"
- config: "config.csv"
- scenarios: "scenarios.csv"
- selected_scenarios: ["baseline", "HighDemand"]
- cpu: 4
- sensitivity: true
```

## Troubleshooting

1. **MCP server not found:**
   - Verify the path in the configuration file is correct
   - Ensure Python is in your PATH
   - Check that `mcp` package is installed

2. **Import errors:**
   - Make sure all EPM dependencies are installed
   - Verify you're in the correct Python environment

3. **GAMS errors:**
   - Ensure GAMS is installed and in your PATH
   - Check that input data files exist in the specified folder

## Testing the MCP Server

You can test the MCP server directly from the command line:

```bash
cd /Users/lucas/Documents/World Bank/Projects/EPM_APPLIED/EPM_main
python tools/mcp_server.py
```

The server will start and wait for MCP protocol messages on stdin/stdout.

