const SecondaryButton = ({
  onClick,
  onMouseEnter,
  onMouseLeave,
  children,
  className = "",
  isActive = true,
}) => {
  // Detect if custom gradient colors (like from-warmyellow / to-darkpink) were passed
  const hasCustomGradient = /from-|to-/.test(className);

  // Default gradient and hover bright variant
  const defaultGradient = isActive
    ? "from-warmyellow/80 to-electricblue/80"
    : "from-warmyellow/70 to-electricblue/70";

  const defaultHover = isActive
    ? "hover:from-warmyellow hover:to-electricblue"
    : "hover:from-warmyellow hover:to-electricblue";

  // Automatically brighten custom gradients on hover
  const customHover = hasCustomGradient
    ? className
        .replace(/from-([^\s]+)/, "hover:from-$20/100")
        .replace(/to-([^\s]+)/, "hover:to-$20/100")
    : "";

  return (
    <button
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      className={`
        bg-radial rounded-full px-4 py-2 text-sm font-semibold text-blueblack transition cursor-pointer
        ${hasCustomGradient ? "" : `${defaultGradient} ${defaultHover}`}
        ${customHover}
        ${className}
      `}
    >
      {children}
    </button>
  );
};

export default SecondaryButton;
