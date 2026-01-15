import { createContext, useState, useCallback, useEffect, useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import Joyride, { EVENTS, STATUS, ACTIONS } from 'react-joyride'
import {
  loadTutorialState,
  saveTutorialState,
  resetTutorialState,
  markTourComplete,
  updateTourProgress,
  dismissHint as dismissHintStorage,
  setHintsEnabled as setHintsEnabledStorage,
  setWelcomeSeen
} from '../../utils/tutorialStorage'
import { mainTour } from '../../data/tutorials/mainTour'
import { scenarioBuilderTour } from '../../data/tutorials/scenarioBuilderTour'

export const TutorialContext = createContext(null)

const TOURS = {
  main: mainTour,
  scenarioBuilder: scenarioBuilderTour
}

// Custom tooltip styles matching Tailwind primary color
const tooltipStyles = {
  options: {
    primaryColor: '#2563eb',
    zIndex: 10000
  },
  tooltip: {
    borderRadius: '8px',
    padding: '16px'
  },
  tooltipContainer: {
    textAlign: 'left'
  },
  tooltipTitle: {
    fontSize: '16px',
    fontWeight: '600',
    marginBottom: '8px'
  },
  tooltipContent: {
    fontSize: '14px',
    lineHeight: '1.5'
  },
  buttonNext: {
    backgroundColor: '#2563eb',
    borderRadius: '6px',
    padding: '8px 16px',
    fontSize: '14px',
    fontWeight: '500'
  },
  buttonBack: {
    color: '#6b7280',
    marginRight: '8px'
  },
  buttonSkip: {
    color: '#9ca3af'
  },
  spotlight: {
    borderRadius: '8px'
  }
}

export function TutorialProvider({ children }) {
  const location = useLocation()
  const navigate = useNavigate()

  // Load initial state from localStorage
  const [state, setState] = useState(() => loadTutorialState())

  // Tour state
  const [isRunning, setIsRunning] = useState(false)
  const [currentTourId, setCurrentTourId] = useState(null)
  const [stepIndex, setStepIndex] = useState(0)
  const [steps, setSteps] = useState([])

  // Modal state
  const [showWelcome, setShowWelcome] = useState(false)
  const [showVideoModal, setShowVideoModal] = useState(false)
  const [selectedVideo, setSelectedVideo] = useState(null)

  // Check if first-time user on mount
  useEffect(() => {
    if (!state.hasSeenWelcome && location.pathname === '/') {
      // Small delay to ensure page is rendered
      const timer = setTimeout(() => setShowWelcome(true), 500)
      return () => clearTimeout(timer)
    }
  }, [])

  // Filter steps based on current route for cross-page tours
  const getStepsForCurrentRoute = useCallback((tourId) => {
    const tour = TOURS[tourId]
    if (!tour) return []

    // For now, return all steps - we'll handle navigation in the callback
    return tour.steps
  }, [])

  // Start a tour
  const startTour = useCallback((tourId) => {
    const tour = TOURS[tourId]
    if (!tour) {
      console.warn(`Tour "${tourId}" not found`)
      return
    }

    // Navigate to the start route if specified
    if (tour.startRoute && location.pathname !== tour.startRoute) {
      navigate(tour.startRoute)
    }

    setCurrentTourId(tourId)
    setSteps(tour.steps)
    setStepIndex(0)
    setIsRunning(true)
    setShowWelcome(false)
  }, [location.pathname, navigate])

  // End the current tour
  const endTour = useCallback((completed = false) => {
    if (currentTourId && completed) {
      markTourComplete(currentTourId)
      setState(prev => ({
        ...prev,
        tours: {
          ...prev.tours,
          [currentTourId]: { completed: true, completedAt: new Date().toISOString() }
        }
      }))
    }
    setIsRunning(false)
    setCurrentTourId(null)
    setStepIndex(0)
    setSteps([])
  }, [currentTourId])

  // Skip the current tour
  const skipTour = useCallback(() => {
    endTour(false)
  }, [endTour])

  // Go to a specific step
  const goToStep = useCallback((index) => {
    setStepIndex(index)
  }, [])

  // Toggle hints
  const toggleHints = useCallback(() => {
    const newValue = !state.hintsEnabled
    setHintsEnabledStorage(newValue)
    setState(prev => ({ ...prev, hintsEnabled: newValue }))
  }, [state.hintsEnabled])

  // Dismiss a hint permanently
  const dismissHint = useCallback((hintId) => {
    dismissHintStorage(hintId)
    setState(prev => ({
      ...prev,
      dismissedHints: [...prev.dismissedHints, hintId]
    }))
  }, [])

  // Reset all tutorial progress
  const resetProgress = useCallback(() => {
    const newState = resetTutorialState()
    setState(newState)
    setShowWelcome(true)
  }, [])

  // Close welcome modal
  const closeWelcome = useCallback(() => {
    setWelcomeSeen()
    setState(prev => ({ ...prev, hasSeenWelcome: true }))
    setShowWelcome(false)
  }, [])

  // Open video modal
  const openVideoModal = useCallback((videoId = null) => {
    setSelectedVideo(videoId)
    setShowVideoModal(true)
  }, [])

  // Close video modal
  const closeVideoModal = useCallback(() => {
    setShowVideoModal(false)
    setSelectedVideo(null)
  }, [])

  // Check if a tour is completed
  const isTourCompleted = useCallback((tourId) => {
    return state.tours[tourId]?.completed || false
  }, [state.tours])

  // Joyride callback handler
  const handleJoyrideCallback = useCallback((data) => {
    const { action, index, status, type, step } = data

    // Handle step changes
    if (type === EVENTS.STEP_AFTER || type === EVENTS.TARGET_NOT_FOUND) {
      const nextIndex = index + (action === ACTIONS.PREV ? -1 : 1)

      // Check if we need to navigate to a different page
      if (steps[nextIndex]?.route && steps[nextIndex].route !== location.pathname) {
        setStepIndex(nextIndex)
        navigate(steps[nextIndex].route)
        return
      }

      setStepIndex(nextIndex)

      // Save progress
      if (currentTourId) {
        updateTourProgress(currentTourId, nextIndex)
      }
    }

    // Handle tour completion
    if (status === STATUS.FINISHED) {
      endTour(true)
    }

    // Handle tour skip
    if (status === STATUS.SKIPPED || action === ACTIONS.CLOSE) {
      endTour(false)
    }
  }, [currentTourId, endTour, location.pathname, navigate, steps])

  // Context value
  const value = useMemo(() => ({
    // State
    isRunning,
    currentTourId,
    stepIndex,
    steps,
    hintsEnabled: state.hintsEnabled,
    dismissedHints: state.dismissedHints,
    hasSeenWelcome: state.hasSeenWelcome,
    tours: state.tours,

    // Modal state
    showWelcome,
    showVideoModal,
    selectedVideo,

    // Actions
    startTour,
    endTour,
    skipTour,
    goToStep,
    toggleHints,
    dismissHint,
    resetProgress,
    closeWelcome,
    openVideoModal,
    closeVideoModal,

    // Helpers
    isTourCompleted,
    availableTours: Object.keys(TOURS)
  }), [
    isRunning, currentTourId, stepIndex, steps,
    state.hintsEnabled, state.dismissedHints, state.hasSeenWelcome, state.tours,
    showWelcome, showVideoModal, selectedVideo,
    startTour, endTour, skipTour, goToStep,
    toggleHints, dismissHint, resetProgress,
    closeWelcome, openVideoModal, closeVideoModal,
    isTourCompleted
  ])

  return (
    <TutorialContext.Provider value={value}>
      {children}
      {isRunning && steps.length > 0 && (
        <Joyride
          steps={steps}
          stepIndex={stepIndex}
          run={isRunning}
          continuous
          showProgress
          showSkipButton
          scrollToFirstStep
          disableScrolling={false}
          callback={handleJoyrideCallback}
          styles={tooltipStyles}
          locale={{
            back: 'Back',
            close: 'Close',
            last: 'Finish',
            next: 'Next',
            skip: 'Skip tour'
          }}
        />
      )}
    </TutorialContext.Provider>
  )
}

export default TutorialProvider
