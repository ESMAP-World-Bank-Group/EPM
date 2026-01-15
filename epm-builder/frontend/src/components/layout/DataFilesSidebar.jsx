import { useState } from 'react'
import { useDataFiles } from '../../context/DataFilesContext'
import { getCategoryIcon } from '../icons/CategoryIcons'

function DataFilesSidebar({ onCategoryClick, onRunScenario, isRunDisabled, isSubmitting }) {
  const { schema, loading, getCategoryCounts, getTotalCustomCount, getFileStatus } = useDataFiles()
  const [expandedCategory, setExpandedCategory] = useState(null)
  const [showMvpTooltip, setShowMvpTooltip] = useState(false)

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

  // Calculate category consistency status (green if has files, always green for MVP since template provides defaults)
  const getCategoryStatus = (category) => {
    const count = counts[category] || { total: 0, custom: 0 }
    // For MVP, all categories are "ready" since template data provides defaults
    // Custom files just override the defaults
    return count.total > 0 ? 'ready' : 'empty'
  }

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
          const categoryStatus = getCategoryStatus(category)

          return (
            <div key={category} className="mb-1">
              {/* Category Row */}
              <button
                onClick={() => handleCategoryClick(category)}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors text-left group"
              >
                <div className="flex items-center space-x-2 min-w-0">
                  {/* Status indicator */}
                  <span
                    className={`w-2 h-2 rounded-full flex-shrink-0 ${
                      categoryStatus === 'ready' ? 'bg-green-500' : 'bg-gray-300'
                    }`}
                    title={categoryStatus === 'ready' ? 'Template data available' : 'No data'}
                  />
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

        {/* Run Scenario Button */}
        {onRunScenario && (
          <div className="mt-4 relative">
            <button
              onClick={onRunScenario}
              disabled={true}
              onMouseEnter={() => setShowMvpTooltip(true)}
              onMouseLeave={() => setShowMvpTooltip(false)}
              className="w-full px-4 py-2 bg-gray-400 text-white rounded-md cursor-not-allowed flex items-center justify-center"
            >
              {isSubmitting ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creating...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Run Scenario
                </>
              )}
            </button>
            {/* MVP Tooltip */}
            {showMvpTooltip && (
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 text-white text-xs rounded-lg whitespace-nowrap z-50">
                <div className="flex items-center">
                  <svg className="w-4 h-4 mr-1 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  MVP Version - UI only, EPM model not connected
                </div>
                <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
                  <div className="border-4 border-transparent border-t-gray-800"></div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default DataFilesSidebar
