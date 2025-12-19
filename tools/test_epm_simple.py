#!/usr/bin/env python3
"""
Simple test script to run epm.py directly.
This is an alternative to MCP if you just want to test running EPM.
"""

import sys
import os
from pathlib import Path

# Add epm directory to path so we can import
# Script is in tools/, so go up one level to get project root
TOOLS_DIR = Path(__file__).parent.absolute()
BASE_DIR = TOOLS_DIR.parent.absolute()
EPM_DIR = BASE_DIR / "epm"
sys.path.insert(0, str(EPM_DIR))

# Change to EPM directory for proper working directory
os.chdir(EPM_DIR)

# Now import and call main directly
# Import epm.epm module to get the main function
try:
    import epm.epm as epm_module
    main = epm_module.main
except ImportError:
    # If import fails, we'll handle it in the function
    main = None

def run_epm_simple(folder_input="data_test", cpu=1):
    """Run EPM with simple parameters by calling main() directly."""
    # Build arguments list for main()
    test_args = [
        "--folder_input", folder_input,
        "--cpu", str(cpu)
    ]
    
    print(f"Running EPM with: {' '.join(test_args)}")
    print(f"Working directory: {EPM_DIR}")
    print("-" * 60)
    
    try:
        if main is None:
            raise ImportError("Could not import epm.epm module. Make sure all dependencies are installed.")
        main(test_args=test_args)
        print("-" * 60)
        print("✓ EPM completed successfully!")
        return 0
    except SystemExit as e:
        # argparse can call sys.exit(), which raises SystemExit
        if e.code == 0 or e.code is None:
            print("-" * 60)
            print("✓ EPM completed successfully!")
            return 0
        else:
            print("-" * 60)
            print(f"✗ EPM failed with exit code: {e.code}")
            return e.code or 1
    except Exception as e:
        print("-" * 60)
        print(f"✗ EPM failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple EPM test runner")
    parser.add_argument("--folder_input", default="data_test", help="Input folder")
    parser.add_argument("--cpu", type=int, default=1, help="Number of CPUs")
    
    args = parser.parse_args()
    
    sys.exit(run_epm_simple(args.folder_input, args.cpu))

