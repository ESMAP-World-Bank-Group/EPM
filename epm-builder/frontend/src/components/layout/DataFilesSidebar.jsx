import { useState } from 'react'
import { useDataFiles } from '../../context/DataFilesContext'
import { getCategoryIcon } from '../icons/CategoryIcons'

function DataFilesSidebar({ onCategoryClick }) {
  const { schema, loading, getCategoryCounts, getTotalCustomCount, getFileStatus } = useDataFiles()
  const [expandedCategory, setExpandedCategory] = useState(null)

  if (loading) {
    return (
      <div className="w-64 bg-gray-50 border-r border-gray-200 p-4">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-3 bg-gray-200 rounded w-1/2"></div>
          <div className="h-3 bg-gray-200 rounded w-2/3"></div>
        </div>
      </div>
    )
  }

  if (!schema) {
    return (
      <div className="w-64 bg-gray-50 border-r border-gray-200 p-4">
        <p className="text-sm text-gray-500">Failed to load schema</p>
      </div>
    )
  }

  const counts = getCategoryCounts()
  const totalCustom = getTotalCustomCount()
  const { files: fileSchema, categories: categoryMeta } = schema

  const handleCategoryClick = (category) => {
    if (expandedCategory === category) {
      setExpandedCategory(null)
    } else {
      setExpandedCategory(category)
    }
  }

  const handleCategoryNavigate = (category, e) => {
    e.stopPropagation()
    if (onCategoryClick) {
      onCategoryClick(category)
    }
  }

  return (
    <div className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h3 className="font-semibold text-gray-900 text-sm">Data Files</h3>
        <p className="text-xs text-gray-500 mt-1">Input file status summary</p>
      </div>

      {/* Category List */}
      <div className="flex-1 overflow-y-auto p-2">
        {Object.entries(fileSchema).map(([category, files]) => {
          const meta = categoryMeta?.[category] || { label: category, color: 'gray' }
          const count = counts[category] || { total: 0, custom: 0 }
          const IconComponent = getCategoryIcon(category)
          const isExpanded = expandedCategory === category
          const fileList = Object.entries(files)

          return (
            <div key={category} className="mb-1">
              {/* Category Row */}
              <button
                onClick={() => handleCategoryClick(category)}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors text-left group"
              >
                <div className="flex items-center space-x-2 min-w-0">
                  <IconComponent className="h-4 w-4 text-gray-500 flex-shrink-0" />
                  <span className="text-sm text-gray-700 truncate">{meta.label}</span>
                </div>
                <div className="flex items-center space-x-2">
                  {count.custom > 0 && (
                    <span className="text-xs font-medium text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
                      {count.custom}
                    </span>
                  )}
                  <span className="text-xs text-gray-400">{count.total}</span>
                  <svg
                    className={`h-4 w-4 text-gray-400 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Expanded File List */}
              {isExpanded && (
                <div className="ml-6 mt-1 space-y-0.5">
                  {fileList.map(([filename, fileSpec]) => {
                    const status = getFileStatus(filename)
                    const statusStyles = {
                      default: 'bg-green-500',
                      custom: 'bg-blue-500',
                      generated: 'bg-purple-500',
                    }

                    return (
                      <div
                        key={filename}
                        className="flex items-center space-x-2 px-2 py-1 text-xs rounded hover:bg-gray-100"
                      >
                        <span className={`w-1.5 h-1.5 rounded-full ${statusStyles[status]}`}></span>
                        <span className="text-gray-600 truncate flex-1" title={fileSpec.description}>
                          {filename.replace('.csv', '')}
                        </span>
                      </div>
                    )
                  })}
                  {/* Navigate to step button */}
                  <button
                    onClick={(e) => handleCategoryNavigate(category, e)}
                    className="w-full text-left px-2 py-1 text-xs text-primary-600 hover:text-primary-700 hover:bg-primary-50 rounded flex items-center space-x-1"
                  >
                    <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                    <span>Go to step</span>
                  </button>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Footer Summary */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <div className="flex items-center space-x-2 text-xs">
          {totalCustom > 0 ? (
            <>
              <span className="w-2 h-2 rounded-full bg-blue-500"></span>
              <span className="text-gray-700">
                <strong>{totalCustom}</strong> custom/generated file{totalCustom !== 1 ? 's' : ''}
              </span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              <span className="text-gray-500">Using all default templates</span>
            </>
          )}
        </div>

        {/* Legend */}
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-gray-500">
          <div className="flex items-center space-x-1">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
            <span>Default</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
            <span>Custom</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
            <span>Generated</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DataFilesSidebar
