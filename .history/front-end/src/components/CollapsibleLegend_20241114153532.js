import React, { useState } from "react";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";

const CollapsibleLegend = ({
  highlightedSections,
  selectedHighlightedSections,
  togglePlayingSection,
  handleHighlightedSectionSelect,
  playingSection,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative flex">
      {/* Graph container to align the legend correctly */}
      <div className="flex-grow">{/* Place your graph component here */}</div>

      {/* Collapsible Legend */}
      <div
        className={`absolute top-0 h-full bg-blue-50 rounded-lg border-2 border-blue-300 border-solid transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
        style={{
          width: "200px",
          overflow: "hidden",
          zIndex: 1,
          right: "-200px", // Start from the right edge of the graph container
        }}
      >
        {/* Legend content */}
        <div className="p-4 mt-12">
          <div className="justify-center">
            {highlightedSections.map((section, idx) => (
              <div key={idx} className="flex items-center space-x-2">
                <IconButton
                  onClick={() =>
                    togglePlayingSection(idx, section.start, section.end)
                  }
                  style={{ color: section.color }}
                >
                  {playingSection === idx ? (
                    <PauseCircleIcon />
                  ) : (
                    <PlayCircleIcon />
                  )}
                </IconButton>
                <span className="text-sm text-slate-800">{section.label}</span>
                <input
                  type="checkbox"
                  className={`h-4 w-4 border rounded border-${section.color}-300 text-${section.color}-500 accent-${section.color}-500 focus:ring-blue-500`}
                  checked={selectedHighlightedSections.includes(section.label)}
                  onChange={() => handleHighlightedSectionSelect(section.label)}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tab button outside the legend container */}
      <button
        className="absolute top-0 py-2 px-1 text-lg text-white font-semibold border-none bg-blue-500 rounded-l-lg transition ease-in-out delay-50 bg-opacity-50 hover:bg-opacity-75"
        style={{
          writingMode: "vertical-rl",
          transform: "rotate(180deg)",
          right: isOpen ? "-50px" : "-200px", // Ensure the tab stays attached
          zIndex: 2,
        }}
        onClick={() => setIsOpen(!isOpen)}
      >
        Legend
      </button>
    </div>
  );
};

export default CollapsibleLegend;
