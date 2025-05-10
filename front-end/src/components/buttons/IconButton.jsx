const IconButton = ({
  icon: Icon,
  onClick,
  className = "",
  colorClass = "text-white",
  bgClass = "bg-blue-500",
  sizeClass = "w-8 h-8",
  roundedClass = "rounded-full",
  ariaLabel = "icon button",
}) => (
  <button
    type="button"
    onClick={onClick}
    className={`flex items-center justify-center ${bgClass} ${colorClass} ${sizeClass} ${roundedClass} focus:outline-none ${className}`}
    aria-label={ariaLabel}
  >
    {Icon && <Icon className="w-5 h-5" />}
  </button>
);

export default IconButton;
