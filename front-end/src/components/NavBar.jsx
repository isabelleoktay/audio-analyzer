import { useLocation } from "react-router-dom";
import { useState } from "react";
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
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const isActive = (path) => location.pathname === path;

  const handleTooltipsHover = (enable) => {
    if (tooltipMode !== "active") {
      setTooltipMode(enable ? "global" : "inactive");
    }
  };

  const toggleTooltipsActive = () => {
    setTooltipMode((prev) => (prev === "active" ? "inactive" : "active"));
  };

  const handleFeedback = () => {
    window.open("https://forms.gle/WF8g6WrMVsrokqyK6", "_blank");
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  return (
    <nav className="fixed top-0 left-0 w-full h-16 flex items-center justify-between px-4 lg:px-6 py-4 bg-transparent text-lightgray z-50">
      {/* Desktop Layout */}
      <div className="hidden lg:flex items-center justify-between w-full">
        {/* Left Section - Desktop */}
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

        {/* Center Section - Desktop */}
        <div className="absolute left-1/2 transform -translate-x-1/2">
          <ButtonGroup
            buttons={[
              {
                type: "left",
                to: "/",
                label: "analyzer",
                active: isActive("/"),
              },
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

        {/* Right Section - Desktop */}
        <div className="flex items-center space-x-2">
          <SecondaryButton onClick={handleFeedback}>feedback</SecondaryButton>
        </div>
      </div>

      {/* Mobile Layout */}
      <div className="lg:hidden flex items-center justify-between w-full">
        {/* Mobile Left - Logo or Main Action */}
        <SecondaryButton onClick={handleReset}>reset</SecondaryButton>

        {/* Hamburger Menu Button */}
        <button
          onClick={toggleMobileMenu}
          className="flex flex-col justify-center items-center w-8 h-8 space-y-1 focus:outline-none"
          aria-label="Toggle menu"
        >
          <div
            className={`w-6 h-0.5 bg-lightgray transition-all duration-300 ${
              isMobileMenuOpen ? "rotate-45 translate-y-2" : ""
            }`}
          />
          <div
            className={`w-6 h-0.5 bg-lightgray transition-all duration-300 ${
              isMobileMenuOpen ? "opacity-0" : ""
            }`}
          />
          <div
            className={`w-6 h-0.5 bg-lightgray transition-all duration-300 ${
              isMobileMenuOpen ? "-rotate-45 -translate-y-2" : ""
            }`}
          />
        </button>
      </div>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-blueblack bg-opacity-50 z-40"
          onClick={closeMobileMenu}
        />
      )}

      {/* Mobile Menu Dropdown */}
      <div
        className={`lg:hidden fixed top-16 left-0 right-0 bg-blueblack bg-opacity-95 backdrop-blur-sm border-t border-gray-700 transition-all duration-300 z-40 ${
          isMobileMenuOpen
            ? "opacity-100 translate-y-0"
            : "opacity-0 -translate-y-full pointer-events-none"
        }`}
      >
        <div className="px-4 py-6 space-y-4">
          {/* Navigation Links */}
          <div className="flex justify-center space-y-3">
            <ButtonGroup
              buttons={[
                {
                  type: "left",
                  to: "/",
                  label: "analyzer",
                  active: isActive("/"),
                },
                {
                  type: "center",
                  to: "/how-to-use",
                  label: "how to use",
                  active: isActive("/how-to-use"),
                },
                {
                  type: "right",
                  to: "https://github.com/isabelleoktay/audio-analyzer",
                  label: "github",
                  external: true,
                },
              ]}
            />
          </div>

          {/* Mobile Controls */}
          <div className="pt-4 border-t border-gray-700 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-lightgray">tooltips</span>
              <SecondaryButton
                onClick={toggleTooltipsActive}
                isActive={tooltipMode === "active"}
              >
                {tooltipMode === "active" ? "on" : "off"}
              </SecondaryButton>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-lightgray">uploads</span>
              <SecondaryButton
                onClick={() => setUploadsEnabled(!uploadsEnabled)}
                isActive={uploadsEnabled}
              >
                {uploadsEnabled ? "enabled" : "disabled"}
              </SecondaryButton>
            </div>

            <div className="pt-2">
              <SecondaryButton
                onClick={() => {
                  handleFeedback();
                  closeMobileMenu();
                }}
                className="w-full"
              >
                feedback
              </SecondaryButton>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
