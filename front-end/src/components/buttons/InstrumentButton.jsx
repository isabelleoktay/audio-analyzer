// src/components/buttons/InstrumentButton.jsx
const InstrumentButton = ({ onClick, children, className = "", selected }) => {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-8 py-2 text-lg ${
        selected
          ? "bg-gradient-to-r from-darkpink to-electricblue text-blueblack hover:opacity-90"
          : "bg-lightgray/25 text-lightgray hover:bg-gradient-to-r hover:from-darkpink/50 hover:to-electricblue/50"
      } ${className}`}
    >
      {children}
    </button>
  );
};

export default InstrumentButton;
