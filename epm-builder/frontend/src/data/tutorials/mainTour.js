// Main application tour - covers the full workflow from Home to Results
export const mainTour = {
  id: 'main',
  name: 'Quick Start Tour',
  description: 'Learn the basics of EPM Scenario Builder',
  startRoute: '/',
  steps: [
    // Home page steps
    {
      target: '[data-tutorial="hero"]',
      title: 'Welcome to EPM Scenario Builder',
      content: 'This tool helps you plan electricity system expansions using the World Bank\'s Electricity Planning Model (EPM). Let\'s take a quick tour!',
      placement: 'bottom',
      disableBeacon: true
    },
    {
      target: '[data-tutorial="create-btn"]',
      title: 'Create New Scenarios',
      content: 'Click here to start building a new scenario. You\'ll configure inputs like demand, supply, and economics.',
      placement: 'bottom'
    },
    {
      target: '[data-tutorial="recent-runs"]',
      title: 'Track Your Runs',
      content: 'View your previous scenario runs here. You can check their status and access results.',
      placement: 'top'
    },
    // Builder page steps
    {
      target: '[data-tutorial="sidebar"]',
      title: 'Data Files Sidebar',
      content: 'This sidebar shows all data file categories. Green dots indicate default templates, blue dots show custom uploads, and purple dots mark auto-generated files.',
      placement: 'right',
      route: '/builder'
    },
    {
      target: '[data-tutorial="step-indicator"]',
      title: 'Step Navigation',
      content: 'Navigate through 6 steps to configure your scenario. Click any step number to jump directly to it.',
      placement: 'bottom'
    },
    {
      target: '[data-tutorial="scenario-name"]',
      title: 'Name Your Scenario',
      content: 'Give your scenario a descriptive name like "Base Case 2040" or "High Renewables".',
      placement: 'bottom'
    },
    {
      target: '[data-tutorial="zones-select"]',
      title: 'Select Zones',
      content: 'Choose which geographic zones to include in your analysis. Each zone can have different demand and supply characteristics.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="inline-files"]',
      title: 'Custom Data Files',
      content: 'Upload custom CSV files to override template data, or use the defaults. The sidebar shows which files have been customized.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="demand-form"]',
      title: 'Configure Demand',
      content: 'Set base year energy and peak demand for each zone. The growth rate will project demand into future years. This auto-generates pDemandForecast.csv.',
      placement: 'top',
      route: '/builder'
    },
    {
      target: '[data-tutorial="supply-files"]',
      title: 'Supply Data',
      content: 'Upload generator data, storage parameters, costs, and renewable profiles. The model includes a default technology fleet if you skip this.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="economics-form"]',
      title: 'Economic Parameters',
      content: 'Set the WACC (Weighted Average Cost of Capital), discount rate, and carbon pricing. These affect investment decisions and total system cost.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="features-toggles"]',
      title: 'Model Features',
      content: 'Enable or disable model capabilities like storage, hydrogen production, transmission expansion, and economic retirement of plants.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="review-summary"]',
      title: 'Review & Run',
      content: 'Review your configuration before running. The summary shows all your settings and which files will be used.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="run-button"]',
      title: 'Start the Model',
      content: 'Click "Run Scenario" to start the EPM optimization. You\'ll be taken to a progress page where you can watch the model run.',
      placement: 'top'
    }
  ]
}

export default mainTour
