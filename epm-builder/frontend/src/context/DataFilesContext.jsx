import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { getUploadSchema, listSessionUploads } from '../api/client'

const DataFilesContext = createContext(null)

// Category to step mapping for navigation
const CATEGORY_TO_STEP = {
  general: 0,      // General step
  demand: 1,       // Demand step
  supply_generation: 2,  // Supply step
  supply_storage: 2,
  supply_costs: 2,
  supply_renewables: 2,
  transmission: 2, // Also in Supply
  emissions: 3,    // Economics step
  policy: 4,       // Features step
  reserves: 4,
  hydrogen: 4,
}

export function DataFilesProvider({ children }) {
  const [schema, setSchema] = useState(null)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [generatedFiles, setGeneratedFiles] = useState({}) // { filename: previewData }
  const [sessionId, setSessionId] = useState(null)
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
    } catch (err) {
      setUploadedFiles([])
    }
  }

  // Get file status: 'default' | 'custom' | 'generated'
  const getFileStatus = useCallback((filename) => {
    // Check if uploaded (custom)
    const isUploaded = uploadedFiles.some(f => f.filename === filename)
    if (isUploaded) return 'custom'

    // Check if generated from form
    if (generatedFiles[filename]) return 'generated'

    // Default template
    return 'default'
  }, [uploadedFiles, generatedFiles])

  // Mark a file as generated from form inputs
  const setGeneratedFile = useCallback((filename, previewData) => {
    setGeneratedFiles(prev => ({
      ...prev,
      [filename]: previewData
    }))
  }, [])

  // Remove generated file (user cleared form or uploaded custom)
  const removeGeneratedFile = useCallback((filename) => {
    setGeneratedFiles(prev => {
      const next = { ...prev }
      delete next[filename]
      return next
    })
  }, [])

  // Get counts per category
  const getCategoryCounts = useCallback(() => {
    if (!schema) return {}

    const counts = {}
    const { files: fileSchema } = schema

    Object.entries(fileSchema).forEach(([category, files]) => {
      const fileList = Object.keys(files)
      const customCount = fileList.filter(f => getFileStatus(f) !== 'default').length
      counts[category] = {
        total: fileList.length,
        custom: customCount
      }
    })

    return counts
  }, [schema, getFileStatus])

  // Get total custom/generated file count
  const getTotalCustomCount = useCallback(() => {
    const counts = getCategoryCounts()
    return Object.values(counts).reduce((sum, c) => sum + c.custom, 0)
  }, [getCategoryCounts])

  // Get step for a category
  const getStepForCategory = useCallback((category) => {
    return CATEGORY_TO_STEP[category] ?? 0
  }, [])

  // Handle session creation
  const handleSessionCreate = useCallback((newSessionId) => {
    setSessionId(newSessionId)
  }, [])

  // Refresh uploads list
  const refreshUploads = useCallback(() => {
    loadUploads()
  }, [sessionId])

  const value = {
    // State
    schema,
    uploadedFiles,
    generatedFiles,
    sessionId,
    loading,
    error,

    // Actions
    setSessionId: handleSessionCreate,
    refreshUploads,
    setGeneratedFile,
    removeGeneratedFile,

    // Getters
    getFileStatus,
    getCategoryCounts,
    getTotalCustomCount,
    getStepForCategory,
  }

  return (
    <DataFilesContext.Provider value={value}>
      {children}
    </DataFilesContext.Provider>
  )
}

export function useDataFiles() {
  const context = useContext(DataFilesContext)
  if (!context) {
    throw new Error('useDataFiles must be used within a DataFilesProvider')
  }
  return context
}

export default DataFilesContext
