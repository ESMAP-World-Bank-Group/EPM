import { useState, useEffect, useCallback } from 'react'
import {
  getUploadSchema,
  uploadCSV,
  listSessionUploads,
  deleteUpload,
  previewTemplate,
  previewUpload,
  downloadTemplate,
} from '../../api/client'
import CategorySection from './CategorySection'
import ParameterSection from './ParameterSection'

function DataInputPanel({ sessionId, onSessionCreate, onFilesChange }) {
  const [schema, setSchema] = useState(null)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [expandedCategories, setExpandedCategories] = useState({})
  const [expandedParams, setExpandedParams] = useState({})
  const [previews, setPreviews] = useState({})
  const [previewLoading, setPreviewLoading] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Load schema on mount
  useEffect(() => {
    loadSchema()
  }, [])

  // Load uploads when session changes
  useEffect(() => {
    if (sessionId) {
      loadUploads()
    }
  }, [sessionId])

  const loadSchema = async () => {
    try {
      const response = await getUploadSchema()
      setSchema(response.data)
    } catch (err) {
      setError('Failed to load schema')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadUploads = async () => {
    if (!sessionId) return
    try {
      const response = await listSessionUploads(sessionId)
      setUploadedFiles(response.data.files || [])
      if (onFilesChange) {
        onFilesChange(response.data.files || [])
      }
    } catch (err) {
      // Session might not exist yet
      setUploadedFiles([])
    }
  }

  const getFilepath = (category, filename, fileSpec) => {
    const folder = fileSpec?.folder
    return folder ? `${folder}/${filename}` : filename
  }

  const getFileStatus = (filename) => {
    const uploaded = uploadedFiles.find(f => f.filename === filename)
    return uploaded ? 'custom' : 'default'
  }

  const loadPreview = useCallback(async (filepath, isUploaded = false) => {
    if (previews[filepath] || previewLoading[filepath]) return

    setPreviewLoading(prev => ({ ...prev, [filepath]: true }))

    try {
      let response
      if (isUploaded && sessionId) {
        response = await previewUpload(sessionId, filepath)
      } else {
        response = await previewTemplate(filepath)
      }
      setPreviews(prev => ({ ...prev, [filepath]: response.data }))
    } catch (err) {
      console.error('Failed to load preview:', err)
      setPreviews(prev => ({ ...prev, [filepath]: null }))
    } finally {
      setPreviewLoading(prev => ({ ...prev, [filepath]: false }))
    }
  }, [sessionId, previews, previewLoading])

  const handleToggleCategory = (category) => {
    setExpandedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }))
  }

  const handleToggleParam = (filepath) => {
    const isExpanding = !expandedParams[filepath]
    setExpandedParams(prev => ({
      ...prev,
      [filepath]: isExpanding
    }))

    // Load preview when expanding
    if (isExpanding) {
      const isUploaded = uploadedFiles.some(f => {
        const uploadPath = f.path || f.filename
        return uploadPath === filepath || uploadPath.endsWith(filepath)
      })
      loadPreview(filepath, isUploaded)
    }
  }

  const handleUpload = async (file, filename) => {
    try {
      const response = await uploadCSV(file, sessionId)

      // Update session ID if first upload
      if (!sessionId && response.data.session_id) {
        onSessionCreate(response.data.session_id)
      }

      // Reload uploads
      await loadUploads()

      // Clear and reload preview for this file
      const uploaded = uploadedFiles.find(f => f.filename === filename)
      const filepath = uploaded?.path || filename
      setPreviews(prev => {
        const newPreviews = { ...prev }
        delete newPreviews[filepath]
        return newPreviews
      })

      // Reload preview if expanded
      if (expandedParams[filepath]) {
        loadPreview(filepath, true)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed')
    }
  }

  const handleDownload = async (filepath, filename) => {
    try {
      const response = await downloadTemplate(filepath)
      const blob = new Blob([response.data], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      setError('Download failed')
    }
  }

  const handleRemove = async (filename) => {
    if (!sessionId) return
    try {
      await deleteUpload(sessionId, filename)
      await loadUploads()

      // Clear preview
      const filepath = uploadedFiles.find(f => f.filename === filename)?.path
      if (filepath) {
        setPreviews(prev => {
          const newPreviews = { ...prev }
          delete newPreviews[filepath]
          return newPreviews
        })
      }
    } catch (err) {
      setError('Failed to remove file')
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (!schema) {
    return (
      <div className="text-center py-8 text-red-500">
        Failed to load data schema
      </div>
    )
  }

  const { files: fileSchema, categories: categoryMeta } = schema

  return (
    <div className="space-y-4">
      {/* Info banner */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <svg className="w-5 h-5 text-blue-600 mr-2 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <div className="text-sm text-blue-800">
            <strong>Configure model inputs:</strong> Each section shows the default template data.
            Download the template to see the current values, modify as needed, then upload your custom CSV.
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{error}</p>
          <button onClick={() => setError(null)} className="text-xs text-red-500 underline mt-1">
            Dismiss
          </button>
        </div>
      )}

      {/* Category sections */}
      {Object.entries(fileSchema).map(([category, files]) => {
        const meta = categoryMeta?.[category] || { label: category, color: 'gray' }
        const fileList = Object.entries(files)
        const customCount = fileList.filter(([filename]) => getFileStatus(filename) === 'custom').length

        return (
          <CategorySection
            key={category}
            name={meta.label}
            color={meta.color}
            fileCount={fileList.length}
            customCount={customCount}
            isExpanded={expandedCategories[category]}
            onToggle={() => handleToggleCategory(category)}
          >
            {fileList.map(([filename, fileSpec]) => {
              const filepath = getFilepath(category, filename, fileSpec)
              const status = getFileStatus(filename)
              const isUploaded = status === 'custom'
              const previewKey = isUploaded
                ? uploadedFiles.find(f => f.filename === filename)?.path || filepath
                : filepath

              return (
                <ParameterSection
                  key={filename}
                  name={filename.replace('.csv', '').replace(/^p/, '')}
                  description={fileSpec.description}
                  filename={filename}
                  filepath={filepath}
                  status={status}
                  previewData={previews[previewKey]}
                  previewLoading={previewLoading[previewKey]}
                  isExpanded={expandedParams[filepath]}
                  onToggle={() => handleToggleParam(filepath)}
                  onDownload={handleDownload}
                  onUpload={(file) => handleUpload(file, filename)}
                  onRemove={handleRemove}
                />
              )
            })}
          </CategorySection>
        )
      })}

      {/* Summary */}
      {uploadedFiles.length > 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <svg className="h-5 w-5 text-green-500 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span className="text-sm text-green-800">
              <strong>{uploadedFiles.length}</strong> custom file(s) will override template data
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default DataInputPanel
