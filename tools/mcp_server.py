#!/usr/bin/env python3
"""
MCP Server for EPM (Electricity Planning Model)

This MCP server provides tools to run epm.py with various configurations
directly from Cursor.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Any, Optional
from io import StringIO
import contextlib

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("Error: mcp package not installed. Install with: pip install mcp")
    sys.exit(1)

# Get the base directory (where this script is located)
# Script is in tools/, so go up one level to get project root
TOOLS_DIR = Path(__file__).parent.absolute()
BASE_DIR = TOOLS_DIR.parent.absolute()
EPM_DIR = BASE_DIR / "epm"

# Add EPM directory to path and import main
# We import epm.epm to get the main function directly
sys.path.insert(0, str(EPM_DIR))
try:
    import epm.epm as epm_module
    epm_main = epm_module.main
except ImportError as e:
    # Fallback: if import fails (e.g., missing gams), we'll use subprocess
    epm_main = None
    import subprocess

# Create the MCP server
app = Server("epm-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for running EPM."""
    return [
        Tool(
            name="run_epm",
            description="Run the EPM (Electricity Planning Model) with specified parameters. "
                       "This tool executes epm.py with the provided configuration options.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_input": {
                        "type": "string",
                        "description": "Input folder name containing data files (default: 'data_test')",
                        "default": "data_test"
                    },
                    "config": {
                        "type": "string",
                        "description": "Path to the configuration file from the folder_input (default: 'config.csv')",
                        "default": "config.csv"
                    },
                    "scenarios": {
                        "type": "string",
                        "description": "Scenario file name (optional). If provided, enables scenario analysis.",
                        "default": None
                    },
                    "selected_scenarios": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific scenarios to run (e.g., ['baseline', 'HighDemand'])",
                        "default": None
                    },
                    "modeltype": {
                        "type": "string",
                        "description": "Solver type to use in GAMS (e.g., 'MIP', 'RMIP'). If not provided, uses config default.",
                        "default": None
                    },
                    "cpu": {
                        "type": "integer",
                        "description": "Number of CPU cores to use for parallel execution (default: 1)",
                        "default": 1,
                        "minimum": 1
                    },
                    "sensitivity": {
                        "type": "boolean",
                        "description": "Enable sensitivity analysis (default: False)",
                        "default": False
                    },
                    "montecarlo": {
                        "type": "boolean",
                        "description": "Enable Monte Carlo analysis (default: False)",
                        "default": False
                    },
                    "montecarlo_samples": {
                        "type": "integer",
                        "description": "Number of samples for Monte Carlo analysis (default: 10)",
                        "default": 10,
                        "minimum": 1
                    },
                    "uncertainties": {
                        "type": "string",
                        "description": "Uncertainties file name for Monte Carlo analysis",
                        "default": None
                    },
                    "debug": {
                        "type": "boolean",
                        "description": "Enable verbose DEBUG mode for GAMS (default: False)",
                        "default": False
                    },
                    "simulation_label": {
                        "type": "string",
                        "description": "Custom label for the simulation output folder (optional)",
                        "default": None
                    }
                }
            }
        ),
        Tool(
            name="run_epm_simple",
            description="Run EPM with a simplified configuration (quick test). "
                       "Uses default settings with minimal parameters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_input": {
                        "type": "string",
                        "description": "Input folder name (default: 'data_test')",
                        "default": "data_test"
                    },
                    "cpu": {
                        "type": "integer",
                        "description": "Number of CPU cores (default: 1)",
                        "default": 1
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "run_epm":
        return await run_epm(arguments)
    elif name == "run_epm_simple":
        return await run_epm_simple(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def run_epm(args: dict[str, Any]) -> list[TextContent]:
    """Run EPM with the provided arguments by calling main() directly."""
    
    # Build arguments list for main(test_args=...)
    test_args = []
    
    if args.get("folder_input"):
        test_args.extend(["--folder_input", args["folder_input"]])
    
    if args.get("config"):
        test_args.extend(["--config", args["config"]])
    
    if args.get("scenarios"):
        test_args.extend(["--scenarios", args["scenarios"]])
    
    if args.get("selected_scenarios"):
        test_args.extend(["--selected_scenarios"] + args["selected_scenarios"])
    
    if args.get("modeltype"):
        test_args.extend(["--modeltype", args["modeltype"]])
    
    if args.get("cpu", 1) > 1:
        test_args.extend(["--cpu", str(args["cpu"])])
    
    if args.get("sensitivity", False):
        test_args.append("--sensitivity")
    
    if args.get("montecarlo", False):
        test_args.append("--montecarlo")
        if args.get("montecarlo_samples"):
            test_args.extend(["--montecarlo_samples", str(args["montecarlo_samples"])])
        if args.get("uncertainties"):
            test_args.extend(["--uncertainties", args["uncertainties"]])
    
    if args.get("debug", False):
        test_args.append("--debug")
    
    if args.get("simulation_label"):
        test_args.extend(["--simulation_label", args["simulation_label"]])
    
    # Change to EPM directory for proper working directory
    original_cwd = os.getcwd()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        os.chdir(EPM_DIR)
        
        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        output_lines = []
        output_lines.append(f"Running EPM with: {' '.join(test_args)}")
        output_lines.append(f"Working directory: {EPM_DIR}")
        output_lines.append("-" * 60)
        
        # Redirect output
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        try:
            # Call main() directly if available, otherwise use subprocess
            if epm_main is not None:
                epm_main(test_args=test_args)
                success = True
                error_msg = None
            else:
                # Fallback to subprocess if import failed
                import subprocess
                cmd = [sys.executable, "-m", "epm.epm"] + test_args
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=EPM_DIR)
                stdout_capture.write(result.stdout)
                stderr_capture.write(result.stderr)
                success = (result.returncode == 0)
                error_msg = None if success else f"Process exited with code {result.returncode}"
        except SystemExit as e:
            # argparse can call sys.exit(), which raises SystemExit
            success = (e.code == 0 or e.code is None)
            error_msg = str(e) if e.code != 0 else None
        except Exception as e:
            success = False
            error_msg = str(e)
            import traceback
            error_msg += "\n" + traceback.format_exc()
        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        # Get captured output
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        output_lines.append("-" * 60)
        
        if stdout_text:
            output_lines.append(f"Output:\n{stdout_text}")
        
        if stderr_text:
            output_lines.append(f"Errors:\n{stderr_text}")
        
        if error_msg:
            output_lines.append(f"Error: {error_msg}")
        
        if success:
            output_lines.append("✓ EPM execution completed successfully!")
        else:
            output_lines.append("✗ EPM execution failed. Check the error messages above.")
        
        return [TextContent(type="text", text="\n".join(output_lines))]
    
    finally:
        os.chdir(original_cwd)


async def run_epm_simple(args: dict[str, Any]) -> list[TextContent]:
    """Run EPM with simple/default configuration."""
    simple_args = {
        "folder_input": args.get("folder_input", "data_test"),
        "cpu": args.get("cpu", 1)
    }
    return await run_epm(simple_args)


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

