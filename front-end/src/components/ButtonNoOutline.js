import React from "react";

const ButtonNoOutline = ({
  text,
  handleClick,
  fontSize,
  bgColor,
  bgColorHover,
  textColor,
  textColorHover,
  disabled = false,
  width = "",
}) => {
  if (!fontSize) {
    fontSize = "base";
  }

  if (!bgColor) {
    bgColor = "blue-500";
  }

  if (!bgColorHover) {
    bgColorHover = "blue-400";
  }

  if (!textColor) {
    textColor = "white";
  }

  if (!textColorHover) {
    textColorHover = textColor;
  }

  return (
    <button
      onClick={handleClick}
      className={`ml-auto transition ease-in-out delay-50 text-${fontSize} text-center text-${textColor} hover:text-${textColorHover} border-transparent focus:border-transparent focus:ring-0 focus:outline-none bg-${bgColor} py-1 px-2 outline-none rounded ${width} ${
        disabled
          ? "opacity-50 bg-gray-500 cursor-not-allowed"
          : "hover:opacity-75 hover:cursor-pointer"
      }`}
      disabled={disabled}
    >
      {text}
    </button>
  );
};

export default ButtonNoOutline;
