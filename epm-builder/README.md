# EPM Scenario Builder MVP

A web-based interface for the Electricity Planning Model (EPM), inspired by [TransitionZero's Scenario Builder](https://www.transitionzero.org/products/scenario-builder).

## Features

- **Form-based scenario configuration** - Configure key parameters through an intuitive wizard
- **Real-time job tracking** - Monitor EPM runs with live progress updates
- **Results visualization** - View capacity and generation charts
- **CSV file download** - Export detailed results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                           │
│     Home | ScenarioBuilder | RunStatus | Results            │
└────────────────────────────┬────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                           │
│    /api/scenarios | /api/jobs | /api/results | /api/templates
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   EPM Engine (GAMS)                         │
│         epm.py orchestrator → main.gms → Results           │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- GAMS with valid license (CPLEX solver)
- EPM codebase (this repo)

## Quick Start

### 1. Install Backend Dependencies

```bash
cd epm-builder/backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd epm-builder/frontend
npm install
```

### 3. Start the Backend

```bash
cd epm-builder/backend
uvicorn main:app --reload --port 8000
```

### 4. Start the Frontend (new terminal)

```bash
cd epm-builder/frontend
npm run dev
```

### 5. Open the App

Visit http://localhost:5173 in your browser.

## Usage

1. **Create Scenario** - Click "New Scenario" and configure:
   - General: Name, years, zones
   - Demand: Energy/peak demand and growth rates
   - Supply: Uses template generators (CSV upload coming soon)
   - Economics: WACC, discount rate, VoLL
   - Features: Enable/disable model features

2. **Run Model** - Click "Run Scenario" to start EPM

3. **View Results** - Once complete, view:
   - Capacity by year (stacked bar chart)
   - Generation by year (stacked area chart)
   - Download CSV files

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/templates` | GET | Get zones, technologies, defaults |
| `/api/scenarios` | POST | Create scenario |
| `/api/scenarios/{id}` | GET | Get scenario details |
| `/api/jobs` | POST | Start EPM run |
| `/api/jobs/{id}` | GET | Get job status |
| `/api/results/{id}` | GET | Get parsed results |
| `/api/results/{id}/files` | GET | List result files |
| `/api/results/{id}/download/{file}` | GET | Download CSV |

## Project Structure

```
epm-builder/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt
│   ├── routes/
│   │   ├── scenarios.py     # Scenario CRUD
│   │   ├── jobs.py          # Job management
│   │   ├── results.py       # Results fetching
│   │   └── templates.py     # Template data
│   ├── services/
│   │   ├── data_builder.py  # Build EPM input folders
│   │   └── epm_runner.py    # Execute EPM runs
│   └── models/
│       └── schemas.py       # Pydantic models
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── api/client.js
│       └── pages/
│           ├── Home.jsx
│           ├── ScenarioBuilder.jsx
│           ├── RunStatus.jsx
│           └── Results.jsx
└── README.md
```

## Limitations (MVP)

- No user authentication
- Single scenario runs (no comparison)
- In-memory storage (resets on restart)
- No sensitivity analysis or Monte Carlo
- Basic results visualization

## Future Enhancements

- User authentication
- Database persistence
- Multiple scenario comparison
- Advanced CSV upload for all input files
- Sensitivity and Monte Carlo analysis
- Cloud deployment

## License

World Bank - Internal Use
