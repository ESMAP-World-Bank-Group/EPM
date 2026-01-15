import { useState, useRef, useEffect } from 'react'
import ReactPlayer from 'react-player'
import { useTutorial } from '../../hooks/useTutorial'
import { videos, videoList } from '../../data/tutorials/videoContent'
import { markVideoWatched } from '../../utils/tutorialStorage'

function VideoModal() {
  const { showVideoModal, closeVideoModal, selectedVideo, startTour } = useTutorial()
  const [currentVideoId, setCurrentVideoId] = useState(selectedVideo || 'overview')
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [activeHotspots, setActiveHotspots] = useState([])
  const playerRef = useRef(null)

  const currentVideo = videos[currentVideoId]

  // Update video when selectedVideo changes
  useEffect(() => {
    if (selectedVideo) {
      setCurrentVideoId(selectedVideo)
    }
  }, [selectedVideo])

  // Update active hotspots based on video progress
  useEffect(() => {
    if (currentVideo?.hotspots) {
      const currentTime = progress * currentVideo.duration
      const active = currentVideo.hotspots.filter(
        h => currentTime >= h.startTime && currentTime <= h.endTime
      )
      setActiveHotspots(active)
    }
  }, [progress, currentVideo])

  if (!showVideoModal) return null

  const handleProgress = (state) => {
    setProgress(state.played)
  }

  const handleEnded = () => {
    setPlaying(false)
    markVideoWatched(currentVideoId)
  }

  const handleStartRelatedTour = () => {
    if (currentVideo?.relatedTour) {
      closeVideoModal()
      startTour(currentVideo.relatedTour)
    }
  }

  const handleClose = () => {
    setPlaying(false)
    closeVideoModal()
  }

  // Check if video URL is available
  const hasVideoUrl = currentVideo?.url && currentVideo.url.length > 0

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black bg-opacity-75 transition-opacity" onClick={handleClose} />

      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white rounded-xl shadow-xl max-w-4xl w-full transform transition-all">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b">
            <h2 className="text-lg font-semibold text-gray-900">Video Tutorials</h2>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex">
            {/* Video List Sidebar */}
            <div className="w-64 border-r bg-gray-50 p-4">
              <h3 className="text-xs font-medium text-gray-500 uppercase mb-3">Available Videos</h3>
              <div className="space-y-2">
                {videoList.map((video) => (
                  <button
                    key={video.id}
                    onClick={() => setCurrentVideoId(video.id)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      currentVideoId === video.id
                        ? 'bg-primary-100 border border-primary-200'
                        : 'hover:bg-gray-100'
                    }`}
                  >
                    <div className="flex items-start">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                        currentVideoId === video.id ? 'bg-primary-500 text-white' : 'bg-gray-200 text-gray-500'
                      }`}>
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className={`text-sm font-medium ${
                          currentVideoId === video.id ? 'text-primary-700' : 'text-gray-900'
                        }`}>
                          {video.title}
                        </p>
                        <p className="text-xs text-gray-500 mt-0.5">
                          {Math.floor(video.duration / 60)}:{(video.duration % 60).toString().padStart(2, '0')}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Video Player Area */}
            <div className="flex-1 p-6">
              <div className="mb-4">
                <h3 className="text-lg font-medium text-gray-900">{currentVideo?.title}</h3>
                <p className="text-sm text-gray-500">{currentVideo?.description}</p>
              </div>

              {/* Video Player */}
              <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                {hasVideoUrl ? (
                  <>
                    <ReactPlayer
                      ref={playerRef}
                      url={currentVideo.url}
                      width="100%"
                      height="100%"
                      playing={playing}
                      controls
                      onProgress={handleProgress}
                      onEnded={handleEnded}
                    />

                    {/* Hotspots Overlay */}
                    {activeHotspots.map((hotspot) => (
                      <div
                        key={hotspot.id}
                        className="absolute bg-white bg-opacity-90 rounded-lg px-3 py-2 shadow-lg animate-pulse"
                        style={{
                          left: hotspot.position.x,
                          top: hotspot.position.y,
                          transform: 'translate(-50%, -50%)'
                        }}
                      >
                        <p className="text-sm text-gray-800">{hotspot.content}</p>
                      </div>
                    ))}
                  </>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      <p className="text-gray-400 text-lg">Video Coming Soon</p>
                      <p className="text-gray-500 text-sm mt-2">
                        This tutorial video is being prepared
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Related Tour */}
              {currentVideo?.relatedTour && (
                <div className="mt-4 p-4 bg-primary-50 rounded-lg flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-primary-800">Want hands-on practice?</p>
                    <p className="text-xs text-primary-600">Try the interactive tour for this topic</p>
                  </div>
                  <button
                    onClick={handleStartRelatedTour}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm font-medium"
                  >
                    Start Tour
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default VideoModal
