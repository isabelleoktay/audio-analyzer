// src/components/buttons/TertiaryButton.jsx
const TertiaryButton = ({ onClick, children, className = "" }) => {
  return (
    <button
      onClick={onClick}
      className={`bg-lightgray/25 rounded-full px-4 py-2 text-lightgray transition hover:bg-lightpink/50 ${className}`}
    >
      {children}
    </button>
  );
};

export default TertiaryButton;
