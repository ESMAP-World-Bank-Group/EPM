import { useState, useEffect, useRef } from 'react'
import { useTutorial } from '../../hooks/useTutorial'

// Hint configuration
const HINTS = {
  'scenario-name': {
    target: '[data-tutorial="scenario-name"]',
    message: 'Start by giving your scenario a descriptive name like "Base Case 2040"',
    delay: 45000, // 45 seconds
    placement: 'bottom'
  },
  'zones-empty': {
    target: '[data-tutorial="zones-select"]',
    message: 'Select at least one zone to include in your analysis',
    delay: 30000,
    placement: 'top'
  },
  'sidebar-unused': {
    target: '[data-tutorial="sidebar"]',
    message: 'Use the sidebar to see all data file categories and their upload status',
    delay: 60000,
    placement: 'right'
  },
  'economics-default': {
    target: '[data-tutorial="economics-form"]',
    message: 'The default WACC is 8%. You may want to adjust this for your region.',
    delay: 30000,
    placement: 'top'
  },
  'results-tabs': {
    target: '[data-tutorial="results-tabs"]',
    message: 'Click the tabs to switch between Capacity and Generation views',
    delay: 30000,
    placement: 'bottom'
  }
}

function ContextualHint({ hintId }) {
  const { hintsEnabled, dismissedHints, dismissHint, isRunning } = useTutorial()
  const [isVisible, setIsVisible] = useState(false)
  const [position, setPosition] = useState({ top: 0, left: 0 })
  const timerRef = useRef(null)
  const hintRef = useRef(null)

  const hint = HINTS[hintId]

  useEffect(() => {
    // Don't show hints if disabled, already dismissed, or tour is running
    if (!hintsEnabled || dismissedHints.includes(hintId) || isRunning || !hint) {
      setIsVisible(false)
      return
    }

    // Start timer to show hint after delay
    timerRef.current = setTimeout(() => {
      const targetElement = document.querySelector(hint.target)
      if (targetElement) {
        const rect = targetElement.getBoundingClientRect()
        const newPosition = calculatePosition(rect, hint.placement)
        setPosition(newPosition)
        setIsVisible(true)
      }
    }, hint.delay)

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [hintId, hintsEnabled, dismissedHints, isRunning, hint])

  const calculatePosition = (rect, placement) => {
    const offset = 12
    switch (placement) {
      case 'top':
        return {
          top: rect.top - offset,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, -100%)'
        }
      case 'bottom':
        return {
          top: rect.bottom + offset,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, 0)'
        }
      case 'left':
        return {
          top: rect.top + rect.height / 2,
          left: rect.left - offset,
          transform: 'translate(-100%, -50%)'
        }
      case 'right':
        return {
          top: rect.top + rect.height / 2,
          left: rect.right + offset,
          transform: 'translate(0, -50%)'
        }
      default:
        return {
          top: rect.bottom + offset,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, 0)'
        }
    }
  }

  const handleDismiss = () => {
    setIsVisible(false)
    dismissHint(hintId)
  }

  const handleDismissTemporary = () => {
    setIsVisible(false)
  }

  if (!isVisible || !hint) return null

  return (
    <div
      ref={hintRef}
      className="fixed z-40 max-w-xs animate-fade-in"
      style={{
        top: position.top,
        left: position.left,
        transform: position.transform
      }}
    >
      <div className="bg-amber-50 border border-amber-200 rounded-lg shadow-lg p-4">
        {/* Hint icon and message */}
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="w-5 h-5 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <p className="ml-3 text-sm text-amber-800">{hint.message}</p>
        </div>

        {/* Actions */}
        <div className="mt-3 flex justify-end space-x-2">
          <button
            onClick={handleDismissTemporary}
            className="text-xs text-amber-600 hover:text-amber-800"
          >
            Got it
          </button>
          <button
            onClick={handleDismiss}
            className="text-xs text-amber-500 hover:text-amber-700"
          >
            Don't show again
          </button>
        </div>
      </div>

      {/* Arrow pointer */}
      <div
        className={`absolute w-3 h-3 bg-amber-50 border-amber-200 transform rotate-45 ${
          hint.placement === 'top' ? 'bottom-[-6px] left-1/2 -translate-x-1/2 border-r border-b' :
          hint.placement === 'bottom' ? 'top-[-6px] left-1/2 -translate-x-1/2 border-l border-t' :
          hint.placement === 'left' ? 'right-[-6px] top-1/2 -translate-y-1/2 border-t border-r' :
          'left-[-6px] top-1/2 -translate-y-1/2 border-b border-l'
        }`}
      />
    </div>
  )
}

// Container component that renders hints based on current page
export function ContextualHintContainer() {
  const { hintsEnabled, isRunning } = useTutorial()

  if (!hintsEnabled || isRunning) return null

  // Detect which hints should be active based on visible elements
  const activeHints = Object.keys(HINTS).filter(hintId => {
    const hint = HINTS[hintId]
    return document.querySelector(hint.target)
  })

  return (
    <>
      {activeHints.map(hintId => (
        <ContextualHint key={hintId} hintId={hintId} />
      ))}
    </>
  )
}

export default ContextualHint
