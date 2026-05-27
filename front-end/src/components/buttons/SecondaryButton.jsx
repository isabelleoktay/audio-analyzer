import React, { useState } from "react";

const SecondaryButton = ({
  onClick,
  onMouseEnter,
  onMouseLeave,
  children,
  className = "",
  disabled = false,
  fromColor: propFromColor,
  toColor: propToColor,
  hoverFromColor: propHoverFromColor,
  hoverToColor: propHoverToColor,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  // Defaults
  const defaultFrom = "from-warmyellow/80";
  const defaultTo = "to-electricblue/80";
  const defaultHoverFrom = "from-warmyellow";
  const defaultHoverTo = "to-electricblue";

  // Allow override via props, then className, then fallback to default
  const fromMatch = className.match(/from-[\w-/]+/);
  const toMatch = className.match(/to-[\w-/]+/);
  const fromColor = propFromColor || (fromMatch ? fromMatch[0] : defaultFrom);
  const toColor = propToColor || (toMatch ? toMatch[0] : defaultTo);
  const hoverFrom =
    propHoverFromColor || fromColor.replace(/\/\d+/, "") || defaultHoverFrom;
  const hoverTo =
    propHoverToColor || toColor.replace(/\/\d+/, "") || defaultHoverTo;

  // Compose className based on hover state
  const gradientClass = disabled
    ? "bg-gray-400 text-gray-700 cursor-not-allowed"
    : isHovered
      ? `bg-radial ${hoverFrom} ${hoverTo}`
      : `bg-radial ${fromColor} ${toColor}`;

  // Internal hover handlers to manage state, but also call external ones
  const handleMouseEnter = (e) => {
    setIsHovered(true);
    if (onMouseEnter) onMouseEnter(e);
  };
  const handleMouseLeave = (e) => {
    setIsHovered(false);
    if (onMouseLeave) onMouseLeave(e);
  };

  return (
    <button
      disabled={disabled}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      className={`
        text-blueblack font-semibold text-sm rounded-full
        px-4 py-2 transition-all duration-200
        ${gradientClass}
        ${className}
      `}
    >
      {children}
    </button>
  );
};

export default SecondaryButton;
