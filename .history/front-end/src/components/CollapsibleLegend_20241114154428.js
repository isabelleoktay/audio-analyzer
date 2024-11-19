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

      {/* Legend with Tab */}
      <div
        className={`absolute top-0 h-full bg-blue-50 border-2 border-blue-300 border-solid transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-[calc(100%-50px)]"
        }`}
        style={{
          width: "250px", // Total width including the tab
          overflow: "hidden",
          zIndex: 1,
        }}
      >
        {/* Legend Content */}
        <div className="flex h-full">
          <div className="flex-1 p-4 mt-12">
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

          {/* Tab button within the legend container */}
          <button
            className="w-12 flex items-center justify-center text-lg text-white font-semibold bg-blue-500 transition ease-in-out bg-opacity-50 hover:bg-opacity-75"
            style={{
              writingMode: "vertical-rl",
              transform: "rotate(180deg)",
              zIndex: 2,
            }}
            onClick={() => setIsOpen(!isOpen)}
          >
            Legend
          </button>
        </div>
      </div>
    </div>
  );
};

export default CollapsibleLegend;
