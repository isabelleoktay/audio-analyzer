import React from "react";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";
import SquareIcon from "@mui/icons-material/Square";

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
    <div className="flex-1 p-4 flex flex-col justify-center space-y-4">
      {sections.map((section, idx) => (
        <div key={idx} className="flex items-center space-x-2">
          {select && (
            <input
              type="checkbox"
              className={`h-5 w-5 border rounded border-${section.color}-300 text-${section.color}-500 accent-${section.color}-500 focus:ring-blue-500`}
              checked={selectedSections.includes(section.label)}
              onChange={() => handleSectionSelect(section.label)}
            />
          )}
          {audio ? (
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
          ) : (
            <SquareIcon style={{ color: section.lineColor }} />
          )}
          <span className="text-sm text-slate-800">{section.label}</span>
        </div>
      ))}
    </div>
  );
};

export default LegendItems;
