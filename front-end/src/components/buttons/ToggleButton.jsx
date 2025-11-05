import React from "react";

const ToggleButton = ({
  options = [],
  selected,
  onChange,
  miniVersion = false,
  background = "bg-white/10",
}) => {
  return (
    <div
      className={`${background} ${
        miniVersion ? "p-2 rounded-xl" : "p-4 rounded-2xl"
      } flex items-center justify-center gap-2 transition-all duration-300`}
    >
      {options.map((option, index) => {
        const isSelected = selected === option;
        return (
          <button
            key={index}
            onClick={() => onChange(option)}
            className={`
              ${
                miniVersion
                  ? "text-xs px-3 py-1 rounded-lg"
                  : "text-sm px-5 py-2 rounded-xl"
              }
              font-semibold transition-all duration-200
              ${
                isSelected
                  ? "bg-radial from-warmyellow/80 to-darkpink/80 text-blueblack shadow-md"
                  : "bg-lightgray/40 text-gray-300 hover:bg-darkgray/40"
              }
            `}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
};

export default ToggleButton;
