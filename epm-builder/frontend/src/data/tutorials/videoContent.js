// Video tutorial definitions with hotspots
export const videos = {
  overview: {
    id: 'overview',
    title: 'EPM Overview',
    description: 'Introduction to EPM User Interface and its capabilities',
    duration: 120, // seconds
    // Placeholder URL - replace with actual video
    url: '',
    thumbnail: null,
    hotspots: [
      {
        id: 'overview-1',
        startTime: 10,
        endTime: 20,
        position: { x: '50%', y: '50%' },
        content: 'This is the main dashboard'
      },
      {
        id: 'overview-2',
        startTime: 45,
        endTime: 55,
        position: { x: '30%', y: '40%' },
        content: 'Create scenarios here'
      }
    ],
    relatedTour: 'main'
  },
  dataFiles: {
    id: 'dataFiles',
    title: 'Working with Data Files',
    description: 'How to prepare and upload CSV files for EPM',
    duration: 180,
    url: '',
    thumbnail: null,
    hotspots: [
      {
        id: 'data-1',
        startTime: 15,
        endTime: 25,
        position: { x: '20%', y: '30%' },
        content: 'Download template files first'
      },
      {
        id: 'data-2',
        startTime: 60,
        endTime: 70,
        position: { x: '70%', y: '50%' },
        content: 'Upload your customized CSV'
      }
    ],
    relatedTour: 'dataSetup'
  },
  results: {
    id: 'results',
    title: 'Understanding Results',
    description: 'How to interpret EPM results and download data',
    duration: 120,
    url: '',
    thumbnail: null,
    hotspots: [
      {
        id: 'results-1',
        startTime: 20,
        endTime: 30,
        position: { x: '50%', y: '40%' },
        content: 'Capacity chart shows installed MW'
      },
      {
        id: 'results-2',
        startTime: 50,
        endTime: 60,
        position: { x: '50%', y: '60%' },
        content: 'Download detailed CSV files here'
      }
    ],
    relatedTour: null
  }
}

export const videoList = Object.values(videos)

export default videos
