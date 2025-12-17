# Transitioning from Excel to CSV

Older models used Excel files for inputs, but the current version of EPM requires CSV-formatted inputs. You will need to convert your existing Excel data accordingly.

## Best practice

Start from an existing csv folder:
- `data_test` for a country-level model. This model was built for a study focused on solar+BESS potential in Gambia.
- `data_test_region` for a regional model. This model was built in the context of the SAPP modeling study exploring the benefits of increased regional integration.

The goal is to fill in the same CSV files using the data from your existing Excel-based model.

> Note: Your Excel model may follow older EPM formats. Some inputs may need to be restructured to match the current CSV specifications. Refer to the Input Description section for updated CSV formatting rules.

The example folders include default techno-economic data (e.g. for plant costs, availability, etc.)

You can reuse these files, but be sure to:
- update the zone definitions to match your model 
- include all zones covered in your case study

## Troubleshooting

- Refer to the `Troubleshooting` section for common conversion issues.
- See Running EPM from GAMS for debugging input processing errors. (GAMS runs are often better at revealing input problems early.)
