# EPM Tools

This directory contains utility scripts and tools for working with EPM.

## Files

- **`mcp_server.py`** - MCP (Model Context Protocol) server that allows Cursor AI to run EPM directly
- **`test_epm_simple.py`** - Simple test script to run EPM with minimal configuration

## MCP Server

The MCP server enables running EPM from within Cursor's AI chat. See the main project documentation for setup instructions.

**Location:** `tools/mcp_server.py`

## Simple Test Script

Quick way to test EPM without MCP:

```bash
python tools/test_epm_simple.py --folder_input data_test --cpu 1
```

