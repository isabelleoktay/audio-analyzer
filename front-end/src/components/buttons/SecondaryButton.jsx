const SecondaryButton = ({
  onClick,
  onMouseEnter,
  onMouseLeave,
  children,
  className = "",
  disabled = false,
}) => {
  const fromMatch = className.match(/from-[\w-/]+/);
  const toMatch = className.match(/to-[\w-/]+/);

  const fromColor = fromMatch ? fromMatch[0] : "from-warmyellow/80";
  const toColor = toMatch ? toMatch[0] : "to-electricblue/80";

  const hoverFrom = fromColor.replace(/\/\d+/, "");
  const hoverTo = toColor.replace(/\/\d+/, "");

  return (
    <button
      disabled={disabled}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      className={`
        text-blueblack font-semibold text-sm rounded-full
        px-4 py-2 transition-all duration-200

        ${
          disabled
            ? "bg-gray-400 text-gray-700 cursor-not-allowed"
            : `bg-radial ${fromColor} ${toColor} hover:${hoverFrom} hover:${hoverTo}`
        }

        ${className}
      `}
    >
      {children}
    </button>
  );
};

export default SecondaryButton;
