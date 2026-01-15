import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getTemplates, createScenario, createJob } from '../api/client'
import DataInputPanel from '../components/forms/DataInputPanel'

const STEPS = ['General', 'Data Files', 'Demand', 'Supply', 'Economics', 'Features', 'Review']

function ScenarioBuilder() {
  const navigate = useNavigate()
  const [currentStep, setCurrentStep] = useState(0)
  const [templates, setTemplates] = useState(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const [uploadSessionId, setUploadSessionId] = useState(null)
  const [uploadedFiles, setUploadedFiles] = useState([])

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    start_year: 2025,
    end_year: 2040,
    zones: [],
    demand: [],
    generators: [],
    economics: {
      wacc: 0.08,
      discount_rate: 0.06,
      voll: 1000
    },
    features: {
      enable_capacity_expansion: true,
      enable_transmission_expansion: false,
      enable_storage: true,
      enable_hydrogen: false,
      apply_carbon_price: false,
      apply_co2_constraint: false,
      enable_economic_retirement: false
    },
    emissions: {
      carbon_price_per_ton: 0,
      annual_co2_limit_mt: null,
      min_renewable_share: 0
    },
    model_type: 'RMIP',
    upload_session_id: null
  })

  useEffect(() => {
    loadTemplates()
  }, [])

  const loadTemplates = async () => {
    try {
      const response = await getTemplates()
      setTemplates(response.data)
      // Set default zones
      if (response.data.zones.length > 0) {
        setFormData(prev => ({
          ...prev,
          zones: response.data.zones.slice(0, 3).map(z => z.code)
        }))
      }
    } catch (error) {
      setError('Failed to load template data')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  const updateFormData = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const updateNestedFormData = (parent, field, value) => {
    setFormData(prev => ({
      ...prev,
      [parent]: {
        ...prev[parent],
        [field]: value
      }
    }))
  }

  const handleUploadSessionCreate = (sessionId) => {
    setUploadSessionId(sessionId)
    setFormData(prev => ({
      ...prev,
      upload_session_id: sessionId
    }))
  }

  const handleUploadedFilesChange = (files) => {
    setUploadedFiles(files)
  }

  const handleSubmit = async () => {
    setSubmitting(true)
    setError(null)

    try {
      // Create scenario
      const scenarioResponse = await createScenario(formData)
      const scenarioId = scenarioResponse.data.id

      // Start job
      const jobResponse = await createJob(scenarioId)
      const jobId = jobResponse.data.id

      // Navigate to status page
      navigate(`/status/${jobId}`)
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to create scenario')
      console.error(error)
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  const renderStepContent = () => {
    switch (currentStep) {
      case 0: // General
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Scenario Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => updateFormData('name', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                placeholder="e.g., Base Case 2040"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={formData.description}
                onChange={(e) => updateFormData('description', e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                placeholder="Brief description of the scenario"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Start Year
                </label>
                <input
                  type="number"
                  value={formData.start_year}
                  onChange={(e) => updateFormData('start_year', parseInt(e.target.value))}
                  min={2020}
                  max={2050}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  End Year
                </label>
                <input
                  type="number"
                  value={formData.end_year}
                  onChange={(e) => updateFormData('end_year', parseInt(e.target.value))}
                  min={2025}
                  max={2060}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Zones to Include
              </label>
              <div className="grid grid-cols-3 gap-2 max-h-48 overflow-y-auto border rounded-md p-3">
                {templates?.zones.map((zone) => (
                  <label key={zone.code} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={formData.zones.includes(zone.code)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          updateFormData('zones', [...formData.zones, zone.code])
                        } else {
                          updateFormData('zones', formData.zones.filter(z => z !== zone.code))
                        }
                      }}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="text-sm text-gray-700">{zone.code}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )

      case 1: // Data Files
        return (
          <DataInputPanel
            sessionId={uploadSessionId}
            onSessionCreate={handleUploadSessionCreate}
            onFilesChange={handleUploadedFilesChange}
          />
        )

      case 2: // Demand
        return (
          <div className="space-y-6">
            <p className="text-sm text-gray-600">
              Configure demand for selected zones. For MVP, we use simplified growth rates.
              You can upload detailed demand data via CSV.
            </p>

            {formData.zones.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                Please select zones in the General step first.
              </div>
            ) : (
              <div className="space-y-4">
                {formData.zones.map((zone) => {
                  const demandEntry = formData.demand.find(d => d.zone === zone) || {
                    zone,
                    base_year_energy_gwh: 1000,
                    base_year_peak_mw: 200,
                    annual_growth_rate: 0.03
                  }

                  const updateDemand = (field, value) => {
                    const newDemand = formData.demand.filter(d => d.zone !== zone)
                    newDemand.push({ ...demandEntry, [field]: value })
                    updateFormData('demand', newDemand)
                  }

                  return (
                    <div key={zone} className="border rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">{zone}</h4>
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">
                            Base Energy (GWh)
                          </label>
                          <input
                            type="number"
                            value={demandEntry.base_year_energy_gwh}
                            onChange={(e) => updateDemand('base_year_energy_gwh', parseFloat(e.target.value))}
                            className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">
                            Base Peak (MW)
                          </label>
                          <input
                            type="number"
                            value={demandEntry.base_year_peak_mw}
                            onChange={(e) => updateDemand('base_year_peak_mw', parseFloat(e.target.value))}
                            className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500 mb-1">
                            Growth Rate (%)
                          </label>
                          <input
                            type="number"
                            value={(demandEntry.annual_growth_rate * 100).toFixed(1)}
                            onChange={(e) => updateDemand('annual_growth_rate', parseFloat(e.target.value) / 100)}
                            step="0.1"
                            className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                          />
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )

      case 3: // Supply
        return (
          <div className="space-y-6">
            <p className="text-sm text-gray-600">
              The model includes the existing generator fleet from the template data.
              You can upload custom generator data in the "Data Files" step.
            </p>

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-2">Available Technologies</h4>
              <div className="flex flex-wrap gap-2">
                {templates?.technologies.slice(0, 15).map((tech) => (
                  <span
                    key={tech.code}
                    className={`px-2 py-1 rounded text-xs ${
                      tech.is_renewable
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-700'
                    }`}
                  >
                    {tech.code}
                  </span>
                ))}
              </div>
            </div>

            {uploadedFiles.some(f => f.category === 'supply') ? (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center">
                  <svg className="h-5 w-5 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-sm text-green-800">
                    Custom supply data uploaded: {uploadedFiles.filter(f => f.category === 'supply').map(f => f.filename).join(', ')}
                  </span>
                </div>
              </div>
            ) : (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mt-2 text-sm text-gray-600">
                  No custom supply data uploaded
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  The model will use the template generator fleet
                </p>
              </div>
            )}
          </div>
        )

      case 4: // Economics
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  WACC (%)
                </label>
                <input
                  type="number"
                  value={(formData.economics.wacc * 100).toFixed(1)}
                  onChange={(e) => updateNestedFormData('economics', 'wacc', parseFloat(e.target.value) / 100)}
                  step="0.5"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">Weighted Average Cost of Capital</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Discount Rate (%)
                </label>
                <input
                  type="number"
                  value={(formData.economics.discount_rate * 100).toFixed(1)}
                  onChange={(e) => updateNestedFormData('economics', 'discount_rate', parseFloat(e.target.value) / 100)}
                  step="0.5"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">For NPV calculations</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Value of Lost Load ($/MWh)
                </label>
                <input
                  type="number"
                  value={formData.economics.voll}
                  onChange={(e) => updateNestedFormData('economics', 'voll', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">Cost of unserved energy</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Carbon Price ($/tCO2)
                </label>
                <input
                  type="number"
                  value={formData.emissions.carbon_price_per_ton}
                  onChange={(e) => updateNestedFormData('emissions', 'carbon_price_per_ton', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">Set &gt; 0 to enable carbon pricing</p>
              </div>
            </div>
          </div>
        )

      case 5: // Features
        return (
          <div className="space-y-4">
            <p className="text-sm text-gray-600 mb-4">
              Enable or disable model features
            </p>

            {[
              { key: 'enable_capacity_expansion', label: 'Capacity Expansion', desc: 'Allow new capacity investments' },
              { key: 'enable_transmission_expansion', label: 'Transmission Expansion', desc: 'Allow new interconnection investments' },
              { key: 'enable_storage', label: 'Battery Storage', desc: 'Include grid-scale storage' },
              { key: 'enable_hydrogen', label: 'Hydrogen Production', desc: 'Model hydrogen production' },
              { key: 'apply_carbon_price', label: 'Carbon Pricing', desc: 'Apply carbon price to emissions' },
              { key: 'apply_co2_constraint', label: 'CO2 Constraint', desc: 'Apply emissions cap' },
              { key: 'enable_economic_retirement', label: 'Economic Retirement', desc: 'Allow plant retirement on economic grounds' },
            ].map(({ key, label, desc }) => (
              <label key={key} className="flex items-start space-x-3 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={formData.features[key]}
                  onChange={(e) => updateNestedFormData('features', key, e.target.checked)}
                  className="mt-1 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <div>
                  <div className="font-medium text-gray-900">{label}</div>
                  <div className="text-sm text-gray-500">{desc}</div>
                </div>
              </label>
            ))}
          </div>
        )

      case 6: // Review
        return (
          <div className="space-y-6">
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-3">Scenario Summary</h4>
              <dl className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <dt className="text-gray-500">Name</dt>
                  <dd className="font-medium">{formData.name || '-'}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Planning Horizon</dt>
                  <dd className="font-medium">{formData.start_year} - {formData.end_year}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Zones</dt>
                  <dd className="font-medium">{formData.zones.length} selected</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Model Type</dt>
                  <dd className="font-medium">{formData.model_type}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">WACC</dt>
                  <dd className="font-medium">{(formData.economics.wacc * 100).toFixed(1)}%</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Carbon Price</dt>
                  <dd className="font-medium">${formData.emissions.carbon_price_per_ton}/tCO2</dd>
                </div>
              </dl>
            </div>

            {/* Data Files Summary */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium text-gray-900">Data Files</h4>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="text-sm text-primary-600 hover:text-primary-700 font-medium flex items-center"
                >
                  <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                  </svg>
                  Edit
                </button>
              </div>

              {uploadedFiles.length > 0 ? (
                <div className="space-y-2">
                  <div className="flex items-center text-sm">
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 mr-2">
                      <span className="w-1.5 h-1.5 rounded-full mr-1.5 bg-blue-500"></span>
                      {uploadedFiles.length} Custom
                    </span>
                    <span className="text-gray-500">files will override template data</span>
                  </div>
                  <ul className="text-sm text-gray-600 space-y-1 mt-2">
                    {uploadedFiles.map((file) => (
                      <li key={file.filename} className="flex items-center">
                        <svg className="h-4 w-4 text-blue-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <span className="font-mono text-xs">{file.filename}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div className="flex items-center text-sm">
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700 mr-2">
                    <span className="w-1.5 h-1.5 rounded-full mr-1.5 bg-green-500"></span>
                    Default
                  </span>
                  <span className="text-gray-500">Using template data for all inputs</span>
                </div>
              )}
            </div>

            {/* Features enabled */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-3">Enabled Features</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(formData.features)
                  .filter(([, enabled]) => enabled)
                  .map(([key]) => (
                    <span
                      key={key}
                      className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                    >
                      {key.replace(/^enable_|^apply_/g, '').replace(/_/g, ' ')}
                    </span>
                  ))}
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex">
                <svg className="w-5 h-5 text-blue-600 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <div className="text-sm text-blue-800">
                  <strong>Ready to run.</strong> The scenario will be created and EPM will start
                  optimizing. This may take several minutes depending on the model size.
                </div>
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
                {error}
              </div>
            )}
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="max-w-3xl mx-auto">
      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {STEPS.map((step, index) => (
            <div key={step} className="flex items-center">
              <button
                onClick={() => setCurrentStep(index)}
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                  index === currentStep
                    ? 'bg-primary-600 text-white'
                    : index < currentStep
                    ? 'bg-primary-100 text-primary-600'
                    : 'bg-gray-100 text-gray-400'
                }`}
              >
                {index + 1}
              </button>
              <span className={`ml-2 text-sm hidden sm:inline ${
                index === currentStep ? 'text-primary-600 font-medium' : 'text-gray-500'
              }`}>
                {step}
              </span>
              {index < STEPS.length - 1 && (
                <div className={`w-8 sm:w-16 h-0.5 mx-2 ${
                  index < currentStep ? 'bg-primary-300' : 'bg-gray-200'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Form Content */}
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">{STEPS[currentStep]}</h2>
        {renderStepContent()}
      </div>

      {/* Navigation */}
      <div className="flex justify-between">
        <button
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>

        {currentStep < STEPS.length - 1 ? (
          <button
            onClick={() => setCurrentStep(currentStep + 1)}
            className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            Next
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={submitting || !formData.name}
            className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {submitting ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Creating...
              </>
            ) : (
              'Run Scenario'
            )}
          </button>
        )}
      </div>
    </div>
  )
}

export default ScenarioBuilder
