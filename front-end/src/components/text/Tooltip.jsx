import { useState } from "react";

const Tooltip = ({
  children,
  text,
  show = false,
  position = "top",
  tooltipMode,
  className = "",
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const getPositionClasses = () => {
    switch (position) {
      case "bottom":
        return "top-full mt-2 left-1/2 transform -translate-x-1/2";
      case "left":
        return "right-full mr-2 top-1/2 transform -translate-y-1/2";
      case "right":
        return "left-full ml-2 top-1/2 transform -translate-y-1/2";
      case "top":
      default:
        return "bottom-full mb-2 left-1/2 transform -translate-x-1/2";
    }
  };

  const getTriangleClasses = () => {
    switch (position) {
      case "bottom":
        return "border-l-transparent border-r-transparent border-t-warmyellow/25 border-b-transparent";
      case "left":
        return "border-t-transparent border-b-transparent border-r-transparent border-l-warmyellow/25";
      case "right":
        return "border-t-transparent border-b-transparent border-l-transparent border-r-warmyellow/25";
      case "top":
      default:
        return "border-l-transparent border-r-transparent border-b-transparent border-t-warmyellow/25";
    }
  };

  const getTrianglePositionStyle = () => {
    switch (position) {
      case "top":
        return { top: "100%", left: "50%", transform: "translateX(-50%)" };
      case "bottom":
        return {
          bottom: "100%",
          left: "50%",
          transform: "translateX(-50%) rotate(180deg)",
        };
      case "left":
        return { right: "100%", top: "50%", transform: "translateY(-50%)" };
      case "right":
        return { left: "100%", top: "50%", transform: "translateY(-50%)" };
      default:
        return {};
    }
  };

  // Determine whether to show the tooltip:
  // It should display if "show" is true (global mode)
  // or if tooltipMode is "active" and the user is hovering over the child.
  const shouldShow = show || (tooltipMode === "active" && isHovered);

  return (
    <div
      className={`relative inline-block ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children}
      <div
        className={`absolute text-center ${getPositionClasses()} text-lightgray z-10 transition-opacity duration-200 ease-in-out ${
          shouldShow ? "opacity-100" : "opacity-0"
        }`}
      >
        <div className="bg-warmyellow/25 text-xs rounded-full px-2 py-1 shadow-lg whitespace-nowrap">
          {text}
        </div>
        <div
          className={`absolute w-0 h-0 border-l-4 border-r-4 border-t-4 ${getTriangleClasses()}`}
          style={getTrianglePositionStyle()}
        ></div>
      </div>
    </div>
  );
};

export default Tooltip;
