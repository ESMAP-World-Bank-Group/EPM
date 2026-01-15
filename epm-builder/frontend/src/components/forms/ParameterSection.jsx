import { useState, useRef } from 'react'

const STATUS_STYLES = {
  none: { bg: 'bg-gray-100', text: 'text-gray-600', dot: 'bg-gray-400' },
  default: { bg: 'bg-green-100', text: 'text-green-700', dot: 'bg-green-500' },
  custom: { bg: 'bg-blue-100', text: 'text-blue-700', dot: 'bg-blue-500' },
}

function ParameterSection({
  name,
  description,
  filename,
  filepath,
  status = 'default',
  previewData,
  previewLoading,
  onDownload,
  onUpload,
  onRemove,
  isExpanded,
  onToggle,
}) {
  const fileInputRef = useRef(null)
  const [uploading, setUploading] = useState(false)

  const statusStyle = STATUS_STYLES[status] || STATUS_STYLES.none

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploading(true)
      try {
        await onUpload(file, filename)
      } finally {
        setUploading(false)
      }
    }
    e.target.value = ''
  }

  const handleDownloadClick = () => {
    onDownload(filepath, filename)
  }

  const statusLabel = status === 'none' ? 'None' : status === 'custom' ? 'Custom' : 'Default'

  return (
    <div className="border-b border-gray-100 last:border-b-0">
      {/* Header row */}
      <button
        onClick={onToggle}
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
          <span className="font-medium text-gray-900">{name}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${statusStyle.bg} ${statusStyle.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${statusStyle.dot}`}></span>
            {statusLabel}
          </span>
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 pb-4 pl-11">
          {/* Description */}
          <p className="text-sm text-gray-500 mb-3">{description}</p>

          {/* Preview table */}
          {previewLoading ? (
            <div className="bg-gray-50 rounded-lg p-4 mb-3 flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-600"></div>
              <span className="ml-2 text-sm text-gray-500">Loading preview...</span>
            </div>
          ) : previewData && previewData.headers?.length > 0 ? (
            <div className="bg-gray-50 rounded-lg p-3 mb-3 overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr>
                    {previewData.headers.map((header, i) => (
                      <th key={i} className="text-left font-medium text-gray-700 px-2 py-1 border-b border-gray-200">
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.data?.map((row, rowIdx) => (
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
              {previewData.has_more && (
                <p className="text-xs text-gray-400 mt-2 text-center">
                  Showing first {previewData.total_preview_rows} rows...
                </p>
              )}
            </div>
          ) : (
            <div className="bg-gray-50 rounded-lg p-4 mb-3 text-center text-sm text-gray-500">
              No preview available
            </div>
          )}

          {/* Action buttons */}
          <div className="flex items-center space-x-2">
            <button
              onClick={handleDownloadClick}
              className="inline-flex items-center px-3 py-1.5 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
            >
              <svg className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Template .csv
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="inline-flex items-center px-3 py-1.5 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              {uploading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-1.5"></div>
                  Uploading...
                </>
              ) : (
                <>
                  <svg className="h-4 w-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  Upload .csv
                </>
              )}
            </button>

            {status === 'custom' && (
              <button
                onClick={() => onRemove(filename)}
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
}

export default ParameterSection
