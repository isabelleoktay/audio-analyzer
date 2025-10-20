const SurveyButton = ({
  onClick,
  children,
  className = "",
  isActive = true,
  isSelected = false,
}) => {
  return (
    <button
      onClick={onClick}
      disabled={!isActive}
      className={`
        rounded-full px-4 py-2 text-sm font-semibold text-blueblack transition-all duration-200 cursor-pointer
        ${
          isActive
            ? isSelected
              ? "bg-darkpink text-blueblack"
              : "bg-lightgray hover:bg-lightpink"
            : "bg-lightgray/40 text-gray-300 cursor-not-allowed"
        }
        ${className}
      `}
    >
      {children}
    </button>
  );
};

export default SurveyButton;
