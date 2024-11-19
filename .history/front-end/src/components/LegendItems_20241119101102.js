import React from "react";
import IconButton from "./IconButton"; // Adjust the import path as needed
import PauseCircleIcon from "./PauseCircleIcon"; // Adjust the import path as needed
import PlayCircleIcon from "./PlayCircleIcon"; // Adjust the import path as needed

const LegendItems = ({
  sections,
  selectedSections,
  handleSectionSelect,
  togglePlayingSection,
  playingSection,
  audio,
}) => {
  return (
    <div className="flex-1 p-4 mt-12">
      {sections.map((section, idx) => (
        <div key={idx} className="flex items-center space-x-2">
          <input
            type="checkbox"
            className={`h-4 w-4 border rounded border-${section.color}-300 text-${section.color}-500 accent-${section.color}-500 focus:ring-blue-500`}
            checked={selectedSections.includes(section.label)}
            onChange={() => handleSectionSelect(section.label)}
          />
          {audio && (
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
          )}
          <span className="text-sm text-slate-800">{section.label}</span>
        </div>
      ))}
    </div>
  );
};

export default LegendItems;
