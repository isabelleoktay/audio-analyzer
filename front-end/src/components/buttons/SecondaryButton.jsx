const SecondaryButton = ({
  onClick,
  onMouseEnter,
  onMouseLeave,
  children,
  className = "",
  isActive = true, // New prop to control active/inactive state
  disabled = false,
}) => {
  return (
    <button
      disabled={disabled}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      className={`bg-radial rounded-full px-4 py-2 text-sm font-semibold text-blueblack transition ${
        disabled
          ? "from-warmyellow/50 to-electricblue/50 cursor-not-allowed opacity-60"
          : isActive
          ? "from-warmyellow to-electricblue hover:opacity-90 cursor-pointer"
          : "from-warmyellow/50 to-electricblue/50 hover:from-warmyellow/75 hover:to-electricblue/75 cursor-pointer"
      } ${className}`}
    >
      {children}
    </button>
  );
};

export default SecondaryButton;
