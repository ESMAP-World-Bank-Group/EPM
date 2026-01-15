import { useState, useRef, useCallback, useEffect } from 'react'
import { useDataFiles } from '../../context/DataFilesContext'
import {
  uploadCSV,
  deleteUpload,
  previewTemplate,
  previewUpload,
  downloadTemplate,
} from '../../api/client'
import { getCategoryIcon } from '../icons/CategoryIcons'

function InlineFileSection({ category, title, showTitle = true }) {
  const {
    schema,
    sessionId,
    setSessionId,
    getFileStatus,
    refreshUploads,
    uploadedFiles,
  } = useDataFiles()

  const [expandedFile, setExpandedFile] = useState(null)
  const [previews, setPreviews] = useState({})
  const [previewLoading, setPreviewLoading] = useState({})
  const [uploading, setUploading] = useState({})
  const [error, setError] = useState(null)
  const fileInputRefs = useRef({})

  if (!schema) return null

  const { files: fileSchema, categories: categoryMeta } = schema
  const categoryFiles = fileSchema[category]
  if (!categoryFiles) return null

  const meta = categoryMeta?.[category] || { label: category, color: 'gray' }
  const IconComponent = getCategoryIcon(category)

  const getFilepath = (filename, fileSpec) => {
    const folder = fileSpec?.folder
    return folder ? `${folder}/${filename}` : filename
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

  const handleToggleFile = (filename, filepath, isUploaded) => {
    if (expandedFile === filename) {
      setExpandedFile(null)
    } else {
      setExpandedFile(filename)
      loadPreview(filepath, isUploaded)
    }
  }

  const handleUpload = async (file, filename, filepath) => {
    setUploading(prev => ({ ...prev, [filename]: true }))
    setError(null)

    try {
      const response = await uploadCSV(file, sessionId, category)

      // Update session ID if first upload
      if (!sessionId && response.data.session_id) {
        setSessionId(response.data.session_id)
      }

      // Refresh uploads list
      refreshUploads()

      // Clear and reload preview
      setPreviews(prev => {
        const newPreviews = { ...prev }
        delete newPreviews[filepath]
        return newPreviews
      })

      // Reload preview if expanded
      if (expandedFile === filename) {
        setTimeout(() => loadPreview(filepath, true), 100)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed')
    } finally {
      setUploading(prev => ({ ...prev, [filename]: false }))
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

  const handleRemove = async (filename, filepath) => {
    if (!sessionId) return
    try {
      await deleteUpload(sessionId, filename)
      refreshUploads()

      // Clear preview
      setPreviews(prev => {
        const newPreviews = { ...prev }
        delete newPreviews[filepath]
        return newPreviews
      })
    } catch (err) {
      setError('Failed to remove file')
    }
  }

  const handleFileChange = (e, filename, filepath) => {
    const file = e.target.files?.[0]
    if (file) {
      handleUpload(file, filename, filepath)
    }
    e.target.value = ''
  }

  const fileEntries = Object.entries(categoryFiles)

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Section Header */}
      {showTitle && (
        <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 flex items-center space-x-2">
          <IconComponent className="h-4 w-4 text-gray-500" />
          <span className="font-medium text-gray-700 text-sm">{title || meta.label}</span>
          <span className="text-xs text-gray-400">({fileEntries.length} files)</span>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2">
          <p className="text-sm text-red-700">{error}</p>
          <button onClick={() => setError(null)} className="text-xs text-red-500 underline">
            Dismiss
          </button>
        </div>
      )}

      {/* File List */}
      <div className="divide-y divide-gray-100">
        {fileEntries.map(([filename, fileSpec]) => {
          const filepath = getFilepath(filename, fileSpec)
          const status = getFileStatus(filename)
          const isUploaded = status === 'custom'
          const isGenerated = status === 'generated'
          const isExpanded = expandedFile === filename
          const previewKey = isUploaded
            ? uploadedFiles.find(f => f.filename === filename)?.path || filepath
            : filepath

          const statusStyles = {
            default: { bg: 'bg-green-100', text: 'text-green-700', dot: 'bg-green-500', label: 'Default' },
            custom: { bg: 'bg-blue-100', text: 'text-blue-700', dot: 'bg-blue-500', label: 'Custom' },
            generated: { bg: 'bg-purple-100', text: 'text-purple-700', dot: 'bg-purple-500', label: 'Generated' },
          }
          const style = statusStyles[status]

          return (
            <div key={filename} className="bg-white">
              {/* File Row */}
              <button
                onClick={() => handleToggleFile(filename, filepath, isUploaded)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors text-left"
              >
                <div className="flex items-center space-x-3">
                  <svg
                    className={`h-4 w-4 text-gray-400 transform transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  <span className="font-medium text-gray-900 text-sm">
                    {filename.replace('.csv', '').replace(/^p/, '')}
                  </span>
                </div>
                <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${style.bg} ${style.text}`}>
                  <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${style.dot}`}></span>
                  {style.label}
                </span>
              </button>

              {/* Expanded Content */}
              {isExpanded && (
                <div className="px-4 pb-4 pl-11 bg-gray-50">
                  <p className="text-sm text-gray-500 mb-3">{fileSpec.description}</p>

                  {/* Preview Table */}
                  {previewLoading[previewKey] ? (
                    <div className="bg-white rounded-lg p-4 mb-3 flex items-center justify-center border">
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-600"></div>
                      <span className="ml-2 text-sm text-gray-500">Loading preview...</span>
                    </div>
                  ) : previews[previewKey] && previews[previewKey].headers?.length > 0 ? (
                    <div className="bg-white rounded-lg p-3 mb-3 overflow-x-auto border">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr>
                            {previews[previewKey].headers.map((header, i) => (
                              <th key={i} className="text-left font-medium text-gray-700 px-2 py-1 border-b border-gray-200">
                                {header}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {previews[previewKey].data?.map((row, rowIdx) => (
                            <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                              {row.map((cell, cellIdx) => (
                                <td key={cellIdx} className="px-2 py-1 text-gray-600 whitespace-nowrap">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {previews[previewKey].has_more && (
                        <p className="text-xs text-gray-400 mt-2 text-center">
                          Showing first {previews[previewKey].total_preview_rows} rows...
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="bg-white rounded-lg p-4 mb-3 text-center text-sm text-gray-500 border">
                      No preview available
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleDownload(filepath, filename)}
                      className="inline-flex items-center px-3 py-1.5 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
                    >
                      <svg className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      Template
                    </button>

                    <input
                      ref={el => fileInputRefs.current[filename] = el}
                      type="file"
                      accept=".csv"
                      onChange={(e) => handleFileChange(e, filename, filepath)}
                      className="hidden"
                    />
                    <button
                      onClick={() => fileInputRefs.current[filename]?.click()}
                      disabled={uploading[filename]}
                      className="inline-flex items-center px-3 py-1.5 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors disabled:opacity-50"
                    >
                      {uploading[filename] ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-1.5"></div>
                          Uploading...
                        </>
                      ) : (
                        <>
                          <svg className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                          </svg>
                          Upload
                        </>
                      )}
                    </button>

                    {status === 'custom' && (
                      <button
                        onClick={() => handleRemove(filename, filepath)}
                        className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-red-600 hover:text-red-700 transition-colors"
                      >
                        <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default InlineFileSection
