const STORAGE_KEY = 'epm_tutorial_state'

const DEFAULT_STATE = {
  hasSeenWelcome: false,
  tours: {},
  hintsEnabled: true,
  dismissedHints: [],
  videos: {}
}

export function loadTutorialState() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      return { ...DEFAULT_STATE, ...JSON.parse(stored) }
    }
  } catch (e) {
    console.warn('Failed to load tutorial state:', e)
  }
  return { ...DEFAULT_STATE }
}

export function saveTutorialState(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch (e) {
    console.warn('Failed to save tutorial state:', e)
  }
}

export function resetTutorialState() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch (e) {
    console.warn('Failed to reset tutorial state:', e)
  }
  return { ...DEFAULT_STATE }
}

export function markTourComplete(tourId) {
  const state = loadTutorialState()
  state.tours[tourId] = {
    ...state.tours[tourId],
    completed: true,
    completedAt: new Date().toISOString()
  }
  saveTutorialState(state)
  return state
}

export function updateTourProgress(tourId, stepIndex) {
  const state = loadTutorialState()
  state.tours[tourId] = {
    ...state.tours[tourId],
    lastStep: stepIndex,
    lastVisit: new Date().toISOString()
  }
  saveTutorialState(state)
  return state
}

export function isDismissedHint(hintId) {
  const state = loadTutorialState()
  return state.dismissedHints.includes(hintId)
}

export function dismissHint(hintId) {
  const state = loadTutorialState()
  if (!state.dismissedHints.includes(hintId)) {
    state.dismissedHints.push(hintId)
    saveTutorialState(state)
  }
  return state
}

export function markVideoWatched(videoId) {
  const state = loadTutorialState()
  state.videos[videoId] = {
    watched: true,
    watchedAt: new Date().toISOString()
  }
  saveTutorialState(state)
  return state
}

export function setHintsEnabled(enabled) {
  const state = loadTutorialState()
  state.hintsEnabled = enabled
  saveTutorialState(state)
  return state
}

export function setWelcomeSeen() {
  const state = loadTutorialState()
  state.hasSeenWelcome = true
  saveTutorialState(state)
  return state
}
