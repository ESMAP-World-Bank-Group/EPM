# Contribute Code

Bug fixes, performance improvements, new features: all contributions are welcome. Before starting, check the [open issues](https://github.com/ESMAP-World-Bank-Group/EPM/issues) to avoid duplicating work, or open a new one to discuss your idea first.

---

## Workflow

All contributions go through a **branch + pull request** workflow:

1. **Fork or clone** the repository if you haven't already.
2. **Create a branch** from the latest `main`:
   ```bash
   git checkout main && git pull
   git checkout -b feature/my-update
   ```
3. **Make your changes**, test them, and commit with a clear message:
   ```bash
   git add .
   git commit -m "Fix: correct pAvailability default for solar"
   ```
4. **Push your branch** to GitHub:
   ```bash
   git push origin feature/my-update
   ```
5. **Open a pull request** on [GitHub](https://github.com/ESMAP-World-Bank-Group/EPM/pulls) targeting `main`, with a short description of what changed and why.

A reviewer from the EPM team will review, comment, and merge the PR once approved.

---

## Pull request guidelines

- **One type of change per PR**: don't mix a bug fix with a new feature
- **Describe your changes**: what was the problem, what did you change, and why
- **Test before submitting**: run the model with your changes and verify outputs are as expected
- **Keep case-study-specific changes in your own branch**: only push to `main` what benefits all users

---

## Adding a new technology

To add a new technology to the model, update the following files:

**1. Supply defaults**, under `supply/`:

- `pAvailabilityDefault.csv`
- `pGenDataInputDefault.csv`
- `pCapexTrajectoriesDefault.csv`

Add the technology in the `main` branch if it is general enough to benefit all users.

**2. Zonal definition**

Default technologies are defined by zone. If no zone-specific data is available, define the same technology for all zones.

**3. Resources and postprocessing**

Add the technology to:

- `PostProcessingStaticTechnologies.csv`
- `Coders.csv`

This ensures figures are generated correctly and prevents exceptions during postprocessing.
