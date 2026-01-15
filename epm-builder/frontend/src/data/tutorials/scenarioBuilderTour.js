// Deep-dive tour for the Scenario Builder page
export const scenarioBuilderTour = {
  id: 'scenarioBuilder',
  name: 'Scenario Builder Guide',
  description: 'Detailed guide for building scenarios',
  startRoute: '/builder',
  steps: [
    {
      target: '[data-tutorial="sidebar"]',
      title: 'Data Files Overview',
      content: 'The sidebar organizes all input files into categories. Each category groups related files that control different aspects of the model.',
      placement: 'right',
      disableBeacon: true
    },
    {
      target: '[data-tutorial="scenario-name"]',
      title: 'Scenario Identification',
      content: 'Choose a clear, descriptive name. Good naming helps track multiple scenarios: "Turkey_HighRE_2040" or "BaseCase_NoCarbonPrice".',
      placement: 'bottom'
    },
    {
      target: '[data-tutorial="zones-select"]',
      title: 'Zone Selection',
      content: 'Zones represent geographic regions in your power system. They can be countries, provinces, or custom areas. Select all zones relevant to your study.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="inline-files"]',
      title: 'File Upload System',
      content: 'Each section shows available files with their status. Click "Download Template" to get the expected format, or "Upload" to provide custom data.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="demand-form"]',
      title: 'Demand Configuration',
      content: 'Enter base year values and growth rate. The model will calculate demand for each planning year using compound growth: Demand(y) = Base * (1 + rate)^years.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="supply-files"]',
      title: 'Generation Fleet',
      content: 'pGenDataInput.csv defines existing and candidate generators. Key columns: technology type, capacity (MW), costs, heat rate, and availability.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="economics-form"]',
      title: 'Financial Parameters',
      content: 'WACC affects annualized capital costs. Discount rate is used for NPV calculations. VoLL (Value of Lost Load) sets the penalty for unserved energy.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="features-toggles"]',
      title: 'Advanced Features',
      content: 'These toggles enable model modules. Capacity expansion allows new investments. Storage and hydrogen add technology options. Carbon pricing adds emission costs.',
      placement: 'top'
    },
    {
      target: '[data-tutorial="review-summary"]',
      title: 'Final Review',
      content: 'Before running, verify all settings. Check enabled features match your study goals. Ensure custom files are uploaded for critical inputs.',
      placement: 'top'
    }
  ]
}

export default scenarioBuilderTour
