import { getCategoryIcon } from '../icons/CategoryIcons'

const COLOR_CLASSES = {
  gray: { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' },
  green: { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200' },
  blue: { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-200' },
  purple: { bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-200' },
  orange: { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-200' },
  indigo: { bg: 'bg-indigo-100', text: 'text-indigo-700', border: 'border-indigo-200' },
  yellow: { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-200' },
  teal: { bg: 'bg-teal-100', text: 'text-teal-700', border: 'border-teal-200' },
}

function CategorySection({
  name,
  color = 'gray',
  icon,  // category key for icon lookup (e.g., 'demand', 'supply_generation')
  fileCount = 0,
  customCount = 0,
  children,
  isExpanded,
  onToggle,
}) {
  const colorClasses = COLOR_CLASSES[color] || COLOR_CLASSES.gray
  const IconComponent = icon ? getCategoryIcon(icon) : null

  return (
    <div className={`border rounded-lg overflow-hidden mb-3 ${colorClasses.border}`}>
      {/* Category header */}
      <button
        onClick={onToggle}
        className={`w-full px-4 py-3 flex items-center justify-between ${colorClasses.bg} hover:opacity-90 transition-opacity text-left`}
      >
        <div className="flex items-center space-x-3">
          {IconComponent ? (
            <IconComponent className={`h-5 w-5 ${colorClasses.text}`} />
          ) : (
            <svg
              className={`h-5 w-5 ${colorClasses.text} transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          )}
          <span className={`font-semibold ${colorClasses.text}`}>{name}</span>
          {IconComponent && (
            <svg
              className={`h-4 w-4 ${colorClasses.text} transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-500">
            {fileCount} file{fileCount !== 1 ? 's' : ''}
          </span>
          {customCount > 0 && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-500 text-white">
              {customCount} custom
            </span>
          )}
        </div>
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="bg-white">
          {children}
        </div>
      )}
    </div>
  )
}

export default CategorySection
