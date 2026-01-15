import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { listJobs } from '../api/client'

function Home() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadJobs()
  }, [])

  const loadJobs = async () => {
    try {
      const response = await listJobs()
      setJobs(response.data)
    } catch (error) {
      console.error('Failed to load jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'running': return 'bg-blue-100 text-blue-800'
      case 'failed': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div>
      {/* Hero Section */}
      <div data-tutorial="hero" className="bg-white rounded-lg shadow-sm p-8 mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          EPM Scenario Builder
        </h1>
        <p className="text-lg text-gray-600 mb-6">
          Plan and analyze electricity system expansion scenarios with the
          Electricity Planning Model (EPM). Configure inputs, run optimizations,
          and visualize results.
        </p>
        <Link
          to="/builder"
          data-tutorial="create-btn"
          className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700"
        >
          Create New Scenario
          <svg className="ml-2 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </Link>
      </div>

      {/* Recent Jobs */}
      <div data-tutorial="recent-runs" className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Runs</h2>

        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <p>No scenario runs yet.</p>
            <p className="mt-2">
              <Link to="/builder" className="text-primary-600 hover:underline">
                Create your first scenario
              </Link>
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50"
              >
                <div>
                  <div className="font-medium text-gray-900">{job.scenario_id}</div>
                  <div className="text-sm text-gray-500">
                    {new Date(job.created_at).toLocaleString()}
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                    {job.status}
                  </span>
                  {job.status === 'completed' ? (
                    <Link
                      to={`/results/${job.id}`}
                      className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                    >
                      View Results
                    </Link>
                  ) : job.status === 'running' ? (
                    <Link
                      to={`/status/${job.id}`}
                      className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                    >
                      View Progress
                    </Link>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default Home
