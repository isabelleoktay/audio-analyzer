// src/components/buttons/TertiaryButton.jsx
const TertiaryButton = ({
  onClick,
  children,
  className = "",
  active = false,
}) => {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-4 py-2 transition ${
        active
          ? "bg-lightpink text-blueblack"
          : "bg-lightgray/25 hover:bg-lightpink/50 text-lightgray"
      } ${className}`}
    >
      {children}
    </button>
  );
};

export default TertiaryButton;
