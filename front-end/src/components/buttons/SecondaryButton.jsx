const SecondaryButton = ({
  onClick,
  onMouseEnter,
  onMouseLeave,
  children,
  className = "",
  isActive = true,
}) => {
  // Detect gradient colors in className (like from-darkpink / to-lightpink)
  const fromMatch = className.match(/from-[\w-/]+/);
  const toMatch = className.match(/to-[\w-/]+/);

  // Extract them or use defaults
  const fromColor = fromMatch ? fromMatch[0] : "from-warmyellow/80";
  const toColor = toMatch ? toMatch[0] : "to-electricblue/80";

  // Construct hover states dynamically (remove transparency if present)
  const hoverFrom = fromColor.replace(/\/\d+/, ""); // e.g. from-warmyellow
  const hoverTo = toColor.replace(/\/\d+/, ""); // e.g. to-electricblue

  return (
    <button
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      className={`
        bg-radial ${fromColor} ${toColor}
        hover:${hoverFrom} hover:${hoverTo}
        text-blueblack font-semibold text-sm rounded-full
        px-4 py-2 transition-all duration-200
        ${isActive ? "" : "opacity-60 cursor-not-allowed"}
        ${className}
      `}
    >
      {children}
    </button>
  );
};

export default SecondaryButton;
