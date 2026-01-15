import { useContext } from 'react'
import { TutorialContext } from '../components/tutorial/TutorialProvider'

export function useTutorial() {
  const context = useContext(TutorialContext)
  if (!context) {
    throw new Error('useTutorial must be used within a TutorialProvider')
  }
  return context
}

export default useTutorial
