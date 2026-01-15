import { Routes, Route, Link } from 'react-router-dom'
import Home from './pages/Home'
import ScenarioBuilder from './pages/ScenarioBuilder'
import RunStatus from './pages/RunStatus'
import Results from './pages/Results'
import { TutorialProvider } from './components/tutorial/TutorialProvider'
import TutorialButton from './components/tutorial/TutorialButton'
import WelcomeModal from './components/tutorial/WelcomeModal'
import VideoModal from './components/tutorial/VideoModal'
import { ContextualHintContainer } from './components/tutorial/ContextualHint'

function App() {
  return (
    <TutorialProvider>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <Link to="/" data-tutorial="nav-logo" className="flex items-center">
                  <span className="text-xl font-bold text-primary-600">EPM</span>
                  <span className="ml-2 text-gray-600">User Interface</span>
                </Link>
              </div>
              <div className="flex items-center space-x-4">
                <Link
                  to="/"
                  className="text-gray-600 hover:text-primary-600 px-3 py-2 text-sm font-medium"
                >
                  Home
                </Link>
                <Link
                  to="/builder"
                  data-tutorial="new-scenario"
                  className="bg-primary-600 text-white hover:bg-primary-700 px-4 py-2 rounded-md text-sm font-medium"
                >
                  New Scenario
                </Link>
                <TutorialButton />
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/builder" element={<ScenarioBuilder />} />
            <Route path="/status/:jobId" element={<RunStatus />} />
            <Route path="/results/:jobId" element={<Results />} />
          </Routes>
        </main>

        {/* Tutorial Components */}
        <WelcomeModal />
        <VideoModal />
        <ContextualHintContainer />
      </div>
    </TutorialProvider>
  )
}

export default App
