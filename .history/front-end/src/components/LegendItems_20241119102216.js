import React from "react";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";

const LegendItems = ({
  sections,
  selectedSections,
  handleSectionSelect,
  togglePlayingSection,
  playingSection,
  audio,
  select,
}) => {
  return (
    <div className="flex-1 p-4 mt-12">
      {sections.map((section, idx) => (
        <div key={idx} className="flex items-center space-x-2">
          {select && (
            <input
              type="checkbox"
              className={`h-4 w-4 border rounded border-${section.color}-300 text-${section.color}-500 accent-${section.color}-500 focus:ring-blue-500`}
              checked={selectedSections.includes(section.label)}
              onChange={() => handleSectionSelect(section.label)}
            />
          )}
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
