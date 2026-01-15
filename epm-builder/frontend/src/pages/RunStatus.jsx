import { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { getJob } from '../api/client'

function RunStatus() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadJob()
    const interval = setInterval(loadJob, 2000) // Poll every 2 seconds
    return () => clearInterval(interval)
  }, [jobId])

  const loadJob = async () => {
    try {
      const response = await getJob(jobId)
      setJob(response.data)

      // Navigate to results when completed
      if (response.data.status === 'completed') {
        setTimeout(() => navigate(`/results/${jobId}`), 1000)
      }
    } catch (error) {
      console.error('Failed to load job:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-gray-500">Loading...</div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Job not found</p>
        <Link to="/" className="text-primary-600 hover:underline mt-2 inline-block">
          Back to Home
        </Link>
      </div>
    )
  }

  const getStatusColor = () => {
    switch (job.status) {
      case 'completed': return 'text-green-600'
      case 'running': return 'text-blue-600'
      case 'failed': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getStatusIcon = () => {
    switch (job.status) {
      case 'completed':
        return (
          <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )
      case 'running':
        return (
          <svg className="w-6 h-6 text-blue-600 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        )
      case 'failed':
        return (
          <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        )
      default:
        return (
          <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
    }
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            {getStatusIcon()}
            <div className="ml-3">
              <h2 className="text-xl font-semibold text-gray-900">
                {job.status === 'running' ? 'Running EPM...' :
                 job.status === 'completed' ? 'Completed!' :
                 job.status === 'failed' ? 'Run Failed' : 'Pending'}
              </h2>
              <p className="text-sm text-gray-500">Scenario: {job.scenario_id}</p>
            </div>
          </div>
          <span className={`text-lg font-medium ${getStatusColor()}`}>
            {job.progress_pct}%
          </span>
        </div>

        {/* Progress Bar */}
        <div data-tutorial="progress-bar" className="mb-6">
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${
                job.status === 'completed' ? 'bg-green-500' :
                job.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'
              }`}
              style={{ width: `${job.progress_pct}%` }}
            />
          </div>
        </div>

        {/* Error Message */}
        {job.status === 'failed' && job.error_message && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <h4 className="text-sm font-medium text-red-800 mb-1">Error</h4>
            <p className="text-sm text-red-700 font-mono">{job.error_message}</p>
          </div>
        )}

        {/* Logs */}
        <div data-tutorial="logs-panel">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Logs</h3>
          <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
            {job.logs.length === 0 ? (
              <span className="text-gray-500">Waiting for output...</span>
            ) : (
              job.logs.map((log, index) => (
                <div key={index} className="text-gray-300">{log}</div>
              ))
            )}
          </div>
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

        {job.status === 'completed' && (
          <Link
            to={`/results/${jobId}`}
            className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            View Results
          </Link>
        )}
      </div>
    </div>
  )
}

export default RunStatus
