// src/components/buttons/SecondaryButton.jsx
const SecondaryButton = ({ onClick, children, className = "" }) => {
  return (
    <button
      onClick={onClick}
      className={`bg-radial from-warmyellow to-electricblue rounded-full px-4 py-2 text-sm font-semibold text-blueblack transition hover:opacity-90 ${className}`}
    >
      {children}
    </button>
  );
};

export default SecondaryButton;
