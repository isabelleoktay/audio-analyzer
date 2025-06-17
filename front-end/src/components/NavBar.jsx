// src/components/NavBar.jsx
import { useLocation } from "react-router-dom";
import ButtonGroup from "./buttons/ButtonGroup";
import SecondaryButton from "./buttons/SecondaryButton";
import Tooltip from "./text/Tooltip";

const NavBar = ({
  handleReset,
  uploadsEnabled,
  setUploadsEnabled,
  setTooltipMode,
  tooltipMode,
}) => {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  const handleTooltipsHover = (enable) => {
    if (tooltipMode !== "active") {
      setTooltipMode(enable ? "global" : "inactive");
    }
  };

  const toggleTooltipsActive = () => {
    setTooltipMode((prev) => (prev === "active" ? "inactive" : "active"));
    // Optionally also call setEnableTooltips for your child tooltip components.
  };

  const handleFeedback = () => {
    window.open("https://forms.gle/WF8g6WrMVsrokqyK6", "_blank");
  };
  return (
    <nav className="fixed top-0 left-0 w-full h-16 flex items-center justify-between px-6 py-4 bg-transparent text-white z-50">
      {/* Left Section */}
      <div className="flex items-center space-x-2">
        <SecondaryButton onClick={handleReset}>reset</SecondaryButton>
        <Tooltip
          position="bottom"
          text="toggle tooltips"
          show={tooltipMode === "global"}
          tooltipMode={tooltipMode}
        >
          <SecondaryButton
            onMouseEnter={() => handleTooltipsHover(true)}
            onMouseLeave={() => handleTooltipsHover(false)}
            onClick={toggleTooltipsActive}
            isActive={tooltipMode === "active"}
          >
            ?
          </SecondaryButton>
        </Tooltip>
        <SecondaryButton
          onClick={() => setUploadsEnabled(!uploadsEnabled)}
          isActive={uploadsEnabled}
        >{`uploads ${
          uploadsEnabled ? "enabled" : "disabled"
        }`}</SecondaryButton>
      </div>

      {/* Center Section */}
      <div className="absolute left-1/2 transform -translate-x-1/2">
        <ButtonGroup
          buttons={[
            { type: "left", to: "/", label: "analyzer", active: isActive("/") },
            {
              type: "center",
              to: "/how-to-use",
              label: "how to use",
              active: isActive("/how-to-use"),
            },
            {
              type: "right",
              to: "https://github.com/your-repo",
              label: "github",
              external: true,
            },
          ]}
        />
      </div>

      {/* Right Section */}
      <div className="flex items-center space-x-2">
        <SecondaryButton onClick={handleFeedback}>feedback</SecondaryButton>
      </div>
    </nav>
  );
};

export default NavBar;
