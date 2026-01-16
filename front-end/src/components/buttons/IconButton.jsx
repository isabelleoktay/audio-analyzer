const IconButton = ({
  icon: Icon,
  onClick,
  className = "",
  colorClass = "text-white",
  bgClass = "bg-blue-500",
  sizeClass = "w-8 h-8",
  iconSize = "w-5 h-5",
  roundedClass = "rounded-full",
  ariaLabel = "icon button",
  disabled = false,
}) => (
  <button
    type="button"
    onClick={onClick}
    disabled={disabled}
    className={`flex items-center justify-center ${bgClass} ${colorClass} ${sizeClass} ${roundedClass} focus:outline-none ${
      disabled ? "opacity-50 cursor-not-allowed" : ""
    } ${className}`}
    aria-label={ariaLabel}
  >
    {Icon && <Icon className={iconSize} />}
  </button>
);

export default IconButton;
