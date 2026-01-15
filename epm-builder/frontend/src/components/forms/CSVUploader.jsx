import { useState, useEffect, useCallback } from 'react'
import { getUploadSchema, uploadCSV, listSessionUploads, deleteUpload } from '../../api/client'

const CATEGORY_LABELS = {
  supply: { label: 'Supply', color: 'blue', description: 'Generator, storage, and fuel data' },
  load: { label: 'Load', color: 'green', description: 'Demand forecasts and profiles' },
  trade: { label: 'Trade', color: 'purple', description: 'Transmission and trade limits' },
  constraint: { label: 'Constraints', color: 'orange', description: 'Carbon pricing, emissions caps, targets' },
  settings: { label: 'Settings', color: 'gray', description: 'Model configuration' },
}

function CSVUploader({ sessionId, onSessionCreate, onFilesChange }) {
  const [schema, setSchema] = useState(null)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [expandedCategory, setExpandedCategory] = useState(null)

  useEffect(() => {
    loadSchema()
    if (sessionId) {
      loadUploads()
    }
  }, [sessionId])

  const loadSchema = async () => {
    try {
      const response = await getUploadSchema()
      setSchema(response.data)
    } catch (err) {
      console.error('Failed to load schema:', err)
    }
  }

  const loadUploads = async () => {
    if (!sessionId) return
    try {
      const response = await listSessionUploads(sessionId)
      setUploadedFiles(response.data.files)
      if (onFilesChange) {
        onFilesChange(response.data.files)
      }
    } catch (err) {
      // Session might not exist yet
      setUploadedFiles([])
    }
  }

  const handleFileUpload = useCallback(async (files) => {
    setUploading(true)
    setError(null)

    let currentSessionId = sessionId

    for (const file of files) {
      try {
        const response = await uploadCSV(file, currentSessionId)

        // Update session ID if this is the first upload
        if (!currentSessionId && response.data.session_id) {
          currentSessionId = response.data.session_id
          if (onSessionCreate) {
            onSessionCreate(currentSessionId)
          }
        }
      } catch (err) {
        setError(err.response?.data?.detail || `Failed to upload ${file.name}`)
        break
      }
    }

    setUploading(false)

    // Reload uploads list
    if (currentSessionId) {
      const response = await listSessionUploads(currentSessionId)
      setUploadedFiles(response.data.files)
      if (onFilesChange) {
        onFilesChange(response.data.files)
      }
    }
  }, [sessionId, onSessionCreate, onFilesChange])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.csv'))
    if (files.length > 0) {
      handleFileUpload(files)
    }
  }, [handleFileUpload])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  const handleFileInput = (e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) {
      handleFileUpload(files)
    }
    e.target.value = '' // Reset input
  }

  const handleDelete = async (filename) => {
    if (!sessionId) return
    try {
      await deleteUpload(sessionId, filename)
      await loadUploads()
    } catch (err) {
      setError(`Failed to delete ${filename}`)
    }
  }

  const getCategoryColor = (category) => {
    return CATEGORY_LABELS[category]?.color || 'gray'
  }

  const getFilesByCategory = () => {
    const grouped = {}
    for (const file of uploadedFiles) {
      const cat = file.category || 'custom'
      if (!grouped[cat]) grouped[cat] = []
      grouped[cat].push(file)
    }
    return grouped
  }

  return (
    <div className="space-y-6">
      {/* Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${dragOver ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-gray-400'}
          ${uploading ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        <input
          type="file"
          accept=".csv"
          multiple
          onChange={handleFileInput}
          className="hidden"
          id="csv-upload-input"
          disabled={uploading}
        />
        <label htmlFor="csv-upload-input" className="cursor-pointer">
          <div className="space-y-2">
            <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div className="text-gray-600">
              <span className="font-medium text-primary-600 hover:text-primary-500">
                Click to upload
              </span>
              {' '}or drag and drop
            </div>
            <p className="text-xs text-gray-500">CSV files only</p>
          </div>
        </label>
        {uploading && (
          <div className="mt-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600 mx-auto"></div>
            <p className="text-sm text-gray-500 mt-2">Uploading...</p>
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-3">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Valid Files Reference */}
      {schema && (
        <div className="border rounded-lg overflow-hidden">
          <div className="bg-gray-50 px-4 py-3 border-b">
            <h4 className="font-medium text-gray-900">Supported CSV Files</h4>
            <p className="text-sm text-gray-500">Click a category to see available files</p>
          </div>
          <div className="divide-y">
            {Object.entries(CATEGORY_LABELS).map(([key, { label, color, description }]) => (
              <div key={key}>
                <button
                  onClick={() => setExpandedCategory(expandedCategory === key ? null : key)}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-${color}-100 text-${color}-800`}>
                      {label}
                    </span>
                    <span className="text-sm text-gray-600">{description}</span>
                  </div>
                  <svg
                    className={`h-5 w-5 text-gray-400 transform transition-transform ${expandedCategory === key ? 'rotate-180' : ''}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {expandedCategory === key && schema[key] && (
                  <div className="px-4 py-2 bg-gray-50 border-t">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr>
                          <th className="text-left font-medium text-gray-700 py-1">File</th>
                          <th className="text-left font-medium text-gray-700 py-1">Description</th>
                          <th className="text-left font-medium text-gray-700 py-1">Required Columns</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(schema[key]).map(([filename, spec]) => (
                          <tr key={filename} className="border-t border-gray-200">
                            <td className="py-2 font-mono text-xs text-gray-800">{filename}</td>
                            <td className="py-2 text-gray-600">{spec.description}</td>
                            <td className="py-2 font-mono text-xs text-gray-500">{spec.required_columns.join(', ')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <div className="border rounded-lg overflow-hidden">
          <div className="bg-gray-50 px-4 py-3 border-b">
            <h4 className="font-medium text-gray-900">Uploaded Files ({uploadedFiles.length})</h4>
          </div>
          <div className="divide-y">
            {Object.entries(getFilesByCategory()).map(([category, files]) => (
              <div key={category} className="px-4 py-3">
                <div className="flex items-center space-x-2 mb-2">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-${getCategoryColor(category)}-100 text-${getCategoryColor(category)}-800`}>
                    {CATEGORY_LABELS[category]?.label || category}
                  </span>
                </div>
                <ul className="space-y-1">
                  {files.map((file) => (
                    <li key={file.filename} className="flex items-center justify-between py-1">
                      <div className="flex items-center space-x-2">
                        <svg className="h-4 w-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="text-sm font-mono">{file.filename}</span>
                        <span className="text-xs text-gray-400">({(file.size_bytes / 1024).toFixed(1)} KB)</span>
                      </div>
                      <button
                        onClick={() => handleDelete(file.filename)}
                        className="text-red-500 hover:text-red-700 text-sm"
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default CSVUploader
