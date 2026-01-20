# EPM Model Diagnostic

Interactive diagnostic for EPM optimization results. Helps identify why model results don't look right - even when there's no explicit error.

## Environment Setup

**IMPORTANT**: Always use the `esmap_env` conda environment when running Python commands:
```bash
conda run -n esmap_env python <script>
```

Or activate it first:
```bash
conda activate esmap_env
```

## Instructions

You are an expert at debugging GAMS optimization models for electricity planning. This is an **interactive, conversational diagnostic**. The user often has an intuition that something is wrong but may not know exactly what.

**IMPORTANT**: Perform the full diagnostic automatically without asking for permission at each step. Go through all steps and provide comprehensive findings.

### Step 1: Ask for the scenario folder

Use AskUserQuestion to ask:
- "Which scenario folder should I analyze? (Path to folder containing PA.gdx)"

Provide common options like:
- `epm/output/simulations_test/baseline`
- Let user type custom path

### Step 2: Gather context from the user

Before diving into data, **ask the user what they observed**. Use AskUserQuestion:

**Question**: "What made you think something might be wrong? Select any that apply or describe in 'Other':"
- "A variable is always zero when it shouldn't be"
- "Prices seem too high or too low"
- "Storage/renewables not being used as expected"
- "Costs don't look right"
- Other (free text)

**Follow-up question**: "Do you have a specific variable, equation, or generator in mind?"
- Let user specify (e.g., "vStorage for BESS_Angola", "eSOCUpperBound marginal is negative")
- Or "No, please do a general check"

### Step 3: Targeted investigation based on user input

Based on user's answers, prioritize your investigation:

**If storage issue suspected:**
1. Check `vStorage`, `vCapStor`, `vStorInj`, `vPwrOut` levels
2. Check `eSOCUpperBound`, `eStorageHourTransition` marginals
3. Verify storage is in `st(g)` set
4. Check initialization logic in main.gms:810-832

**If price issue suspected:**
1. Check `pPrice` distribution by zone
2. Look for VOLL events (unmet demand)
3. Check binding transmission constraints

**If capacity/dispatch issue:**
1. Check `vCap` vs `vPwrOut` utilization
2. Look for curtailment
3. Check reserve constraints

### Step 4: Read the GDX file

Use Python with gams.transfer to extract specific data:

```python
import gams.transfer as gt
import pandas as pd

container = gt.Container("<path>/PA.gdx")

# Get variable with levels and marginals
var = container.data["<variable_name>"]
df = var.records
print(df)
```

Key columns in GDX records:
- `level`: The solution value
- `marginal`: Shadow price (dual value)
- `lower`, `upper`: Variable bounds

### Step 5: Cross-reference with GAMS source

Read `epm/base.gms` to understand equation definitions. Pay attention to:
- The `$()` conditional domain - this controls when the equation is active
- Related sets like `st(g)`, `FD(q,d,t)`, `fEnableStorage`

### Step 6: Check input data if needed

Look at CSV files in `epm/input/<data_folder>/supply/`:
- `pGenDataInput.csv` - Generator parameters
- `pStorageDataInput.csv` - Storage parameters
- Check `Status`, `StYr`, `RetrYr` fields

### Step 7: Present findings conversationally

Don't just dump data. Explain what you found in plain language:

**Good**: "I found that vStorage is zero for BESS_Angola even though vCapStor shows 200 MWh of capacity. Looking at the equations, this happens because..."

**Bad**: "Here are 50 rows of data..."

### Output Format

```
## What I Found

[Plain language explanation of the issue]

## Why This Happens

[Technical explanation with references to specific equations/code]

## The Fix

[Concrete steps to resolve - be specific about file, line, value changes]

## Want Me To...

[Offer next steps: "Should I check the input data?", "Want me to look at related constraints?"]
```

## Key Diagnostic Patterns

### Storage capacity exists but not used (vCapStor > 0, vStorage = 0)
- Check if `st(g)` set includes the unit (requires tech="Storage" or "STOPV")
- Check `fEnableStorage` flag
- Check initialization: if `StYr = sStartYear`, line 811 condition fails
- Check `RetrYr`: if missing, line 814 condition fails

### Negative marginal on inequality constraint
- Normal! Means constraint is binding
- Large negative value = high shadow price = relaxing this constraint would help a lot

### Variable at zero with non-zero marginal
- Constraint is preventing the variable from being positive
- The marginal tells you "cost" of that constraint

## Arguments

$ARGUMENTS - Optional: path to the scenario folder (if not provided, will ask interactively)
