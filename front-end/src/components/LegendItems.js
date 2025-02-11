import React from "react";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";
import SquareIcon from "@mui/icons-material/Square";
import Checkbox from "@mui/material/Checkbox";
import CheckBoxOutlineBlankIcon from "@mui/icons-material/CheckBoxOutlineBlank";
import CheckBoxIcon from "@mui/icons-material/CheckBox";

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
    <div className="flex-1 p-4 flex flex-col justify-center space-y-4 ml-2">
      {sections &&
        sections.map((section, idx) => (
          <div key={idx} className="flex items-center">
            {select && (
              <Checkbox
                icon={
                  <CheckBoxOutlineBlankIcon style={{ color: section.color }} />
                }
                checkedIcon={<CheckBoxIcon style={{ color: section.color }} />}
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
              <SquareIcon
                style={{ color: section.lineColor }}
                className="mr-2"
              />
            )}
            <div className="flex flex-col">
              <span className="text-sm text-slate-800">{section.label}</span>
              {section.subLabel && (
                <span className="text-sm text-slate-800">
                  {section.subLabel}
                </span>
              )}
            </div>
          </div>
        ))}
    </div>
  );
};

export default LegendItems;
