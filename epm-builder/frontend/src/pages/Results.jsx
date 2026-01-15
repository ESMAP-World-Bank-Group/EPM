import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area
} from 'recharts'
import { getResults, listResultFiles } from '../api/client'

// Color palette for technologies
const TECH_COLORS = {
  'CCGT': '#2563eb',
  'OCGT': '#3b82f6',
  'ST': '#6b7280',
  'ICE': '#4b5563',
  'Nuclear': '#9333ea',
  'Coal': '#374151',
  'ReservoirHydro': '#0891b2',
  'ROR': '#06b6d4',
  'PV': '#f59e0b',
  'OnshoreWind': '#10b981',
  'OffshoreWind': '#059669',
  'Battery': '#8b5cf6',
  'Biomass': '#84cc16',
  'Geothermal': '#dc2626',
  'CSP': '#f97316',
}

const getColor = (tech, index) => {
  return TECH_COLORS[tech] || `hsl(${(index * 37) % 360}, 70%, 50%)`
}

function Results() {
  const { jobId } = useParams()
  const [results, setResults] = useState(null)
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('capacity')

  useEffect(() => {
    loadResults()
  }, [jobId])

  const loadResults = async () => {
    try {
      const [resultsResponse, filesResponse] = await Promise.all([
        getResults(jobId),
        listResultFiles(jobId)
      ])
      setResults(resultsResponse.data)
      setFiles(filesResponse.data.files || [])
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to load results')
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-500">Loading results...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">{error}</p>
        </div>
        <Link to="/" className="text-primary-600 hover:underline mt-4 inline-block">
          Back to Home
        </Link>
      </div>
    )
  }

  // Transform data for charts
  const capacityData = Object.entries(results.capacity_by_year || {}).map(([year, techs]) => ({
    year,
    ...techs
  }))

  const generationData = Object.entries(results.generation_by_year || {}).map(([year, techs]) => ({
    year,
    ...techs
  }))

  // Get all unique technologies
  const allTechs = new Set()
  capacityData.forEach(d => Object.keys(d).filter(k => k !== 'year').forEach(t => allTechs.add(t)))
  generationData.forEach(d => Object.keys(d).filter(k => k !== 'year').forEach(t => allTechs.add(t)))
  const techList = Array.from(allTechs)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div data-tutorial="results-header" className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Results</h1>
            <p className="text-gray-500">Scenario: {results.scenario_name}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-primary-600">
              ${(results.total_cost_musd || 0).toLocaleString()} M
            </div>
            <div className="text-sm text-gray-500">Total System Cost</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div data-tutorial="results-tabs" className="bg-white rounded-lg shadow-sm">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            {['capacity', 'generation', 'files'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-4 text-sm font-medium border-b-2 ${
                  activeTab === tab
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'capacity' && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Installed Capacity by Year</h3>
              {capacityData.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={capacityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis label={{ value: 'MW', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    {techList.map((tech, index) => (
                      <Bar
                        key={tech}
                        dataKey={tech}
                        stackId="a"
                        fill={getColor(tech, index)}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No capacity data available
                </div>
              )}
            </div>
          )}

          {activeTab === 'generation' && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Generation by Year</h3>
              {generationData.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={generationData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis label={{ value: 'GWh', angle: -90, position: 'insideLeft' }} />
                    <Tooltip />
                    <Legend />
                    {techList.map((tech, index) => (
                      <Area
                        key={tech}
                        type="monotone"
                        dataKey={tech}
                        stackId="1"
                        fill={getColor(tech, index)}
                        stroke={getColor(tech, index)}
                      />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No generation data available
                </div>
              )}
            </div>
          )}

          {activeTab === 'files' && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Download Result Files</h3>
              {files.length > 0 ? (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {files.map((file) => (
                    <a
                      key={file.name}
                      href={`/api/results/${jobId}/download/${file.name}`}
                      className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50"
                    >
                      <div className="flex items-center">
                        <svg className="w-5 h-5 text-gray-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <span className="text-sm font-medium text-gray-700">{file.name}</span>
                      </div>
                      <span className="text-xs text-gray-400">{file.size_kb} KB</span>
                    </a>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No result files available
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <div className="flex justify-between">
        <Link
          to="/"
          className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
        >
          Back to Home
        </Link>
        <Link
          to="/builder"
          className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
        >
          New Scenario
        </Link>
      </div>
    </div>
  )
}

export default Results
