# MVP Data Folder

**THIS FOLDER IS FOR MVP DEPLOYMENT ONLY**

## Purpose

This folder contains a minimal copy of EPM template data to make the backend
self-contained for cloud deployment (Koyeb).

## Structure

```
data/
├── README.md              # This file
└── mvp_template/          # Copy of essential EPM template files
    ├── data_test/         # Template input data (zones, settings, etc.)
    └── resources/         # Technology definitions (pTechFuel.csv)
```

## Important Notes

1. **This is a COPY** - The source of truth is `epm/input/data_test/` and `epm/resources/`
2. **For MVP only** - In production, this should be replaced with:
   - A proper database
   - Volume mounts to the actual EPM data
   - Or an API to fetch configuration data
3. **Keep in sync** - If you update the main EPM template, remember to update this copy

## When to Update

Update this folder when:
- Adding new default technologies
- Changing default settings
- Modifying zone configurations

## Future Improvements

- [ ] Replace with database-backed configuration
- [ ] Add API endpoint to sync with main EPM data
- [ ] Implement proper data versioning
