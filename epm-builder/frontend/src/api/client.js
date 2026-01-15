import axios from 'axios'

// MVP: Hardcoded API URL for deployment
// TODO: Use environment variable in production
const API_BASE = import.meta.env.VITE_API_BASE || 'https://compact-nissy-modelling-assistant-3106933b.koyeb.app/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Templates
export const getTemplates = () => api.get('/templates')

// Scenarios
export const createScenario = (data) => api.post('/scenarios', data)
export const getScenario = (id) => api.get(`/scenarios/${id}`)
export const listScenarios = () => api.get('/scenarios')

// Jobs
export const createJob = (scenarioId) => api.post('/jobs', { scenario_id: scenarioId })
export const getJob = (id) => api.get(`/jobs/${id}`)
export const listJobs = () => api.get('/jobs')

// Results
export const getResults = (jobId) => api.get(`/results/${jobId}`)
export const listResultFiles = (jobId) => api.get(`/results/${jobId}/files`)

// Uploads
export const getUploadSchema = () => api.get('/uploads/schema')
export const uploadCSV = (file, sessionId = null, category = null) => {
  const formData = new FormData()
  formData.append('file', file)
  if (sessionId) formData.append('session_id', sessionId)
  if (category) formData.append('category', category)
  return api.post('/uploads', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}
export const listSessionUploads = (sessionId) => api.get(`/uploads/${sessionId}`)
export const deleteUpload = (sessionId, filename) => api.delete(`/uploads/${sessionId}/${filename}`)
export const deleteSession = (sessionId) => api.delete(`/uploads/${sessionId}`)
export const previewUpload = (sessionId, filepath, rows = 5) =>
  api.get(`/uploads/${sessionId}/preview/${filepath}`, { params: { rows } })

// Template downloads and previews
export const downloadTemplate = (filepath) =>
  api.get(`/templates/download/${filepath}`, { responseType: 'blob' })
export const previewTemplate = (filepath, rows = 5) =>
  api.get(`/templates/preview/${filepath}`, { params: { rows } })

export default api
