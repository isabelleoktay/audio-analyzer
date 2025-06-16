const SecondaryButton = ({
  onClick,
  onMouseEnter,
  onMouseLeave,
  children,
  className = "",
  isActive = true, // New prop to control active/inactive state
}) => {
  return (
    <button
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      className={`bg-radial rounded-full px-4 py-2 text-sm font-semibold text-blueblack transition cursor-pointer ${
        isActive
          ? "from-warmyellow to-electricblue hover:opacity-90"
          : "from-warmyellow/50 to-electricblue/50 hover:from-warmyellow/75 hover:to-electricblue/75"
      } ${className}`}
    >
      {children}
    </button>
  );
};

export default SecondaryButton;
