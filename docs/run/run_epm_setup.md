# EPM Code Set Up

> An Integrated Development Environment (IDE) bundles a code editor, debugger, and project tools in one place so you can write, run, and keep track of software faster. VS Code is a lightweight IDE that still lets you tailor the experience for Python, GAMS, or any other stack through extensions.

## 1. Install or Verify Visual Studio Code
- Download the installer that fits your platform

## 2. Install Extensions

Extensions are plug-ins that add language support, debuggers, linters, and integrations to VS Code. You can install them from the marketplace.

### Python (ID: ms-python.python)
- `Ctrl/Cmd+Shift+X` opens the Extensions view.
- Search **Python** by Microsoft → **Install**.
- Accept prompts to add the **Pylance** language server for IntelliSense and linting.

### GAMS Language Support (ID: gams-dev.gams-language)
- In the Extensions view, search **GAMS**.
- Install **GAMS Language Support** for syntax highlighting, snippets, and lint hooks.
- Make sure the `gams` executable is on PATH; otherwise set `"gams.executable"` inside `.vscode/settings.json`.

### OpenAI Codex Assistant (example ID: openai.codex-vscode)
- Locate **OpenAI Codex** in your marketplace or install the supplied `.vsix`.
- Install, then sign in with your OpenAI organization credentials or API key.
- Use inline completions (`Ctrl/Cmd+Enter`) or the Codex chat panel for AI-assisted coding.

## 3. Configure Debugging (`launch.json`)
- `Ctrl/Cmd+Shift+P` → **Debug: Open launch.json** → choose your Python interpreter.

- Replace the generated JSON with:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debugg: Main",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/epm/epm.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/epm",
            "args": [
                "--config", "config.csv",
                "--folder_input", "data_test",
                "--modeltype", "RMIP",
                "--simple",
                "--simulation_label", "simulations_test"
            ]
        },
        {
            "name": "Debugg: Postprocessing",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/epm/epm.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/epm",
            "args": [
                "--postprocess",
                "output/simulations_test",
            ]
        },
    ]
}
```

## 4. Work with Git in VS Code
- VS Code detects Git repositories automatically and shows the current branch in the status bar.
- Use `Ctrl/Cmd+Shift+G` to open the **Source Control** view, where you can stage files, write commit messages, and push/pull without leaving the IDE.
- The **Timeline** and **Diff** views display changes line by line so you can review edits before committing.
- When collaborating, configure remote repositories once; afterwards VS Code handles fetch, pull, and push through the built-in controls or the integrated terminal.

## 5. Why VS Code Works Well for Python + GAMS
- One workspace covers both languages with shared Git, search, and refactoring.
- Integrated terminals run `python`, `pip`, or `gams` without context switching.
- Launch configurations and tasks automate pipelines (solve in GAMS, post-process in Python).
- Extensions supply IntelliSense, linting, testing, and Codex AI help across both stacks.
