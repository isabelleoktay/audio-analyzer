import React, { useState } from "react";
import LegendItems from "./LegendItems";

const CollapsibleLegend = ({
  sections,
  selectedSections,
  togglePlayingSection,
  handleSectionSelect,
  playingSection,
  activeTab,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative flex">
      {/* Graph container to align the legend correctly */}
      <div className="flex-grow">{/* Place your graph component here */}</div>

      {/* Legend with Tab */}
      <div
        className={`absolute top-0 h-full bg-blue-50 rounded-r-lg border-y-2 border-r-2 border-blue-500 border-solid transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-[calc(100%-50px)]"
        }`}
        style={{
          width: "250px", // Total width including the tab
          overflow: "hidden",
          zIndex: 1,
        }}
      >
        {/* Legend Content */}
        <div className="flex h-full relative">
          {/* Vertical line divider */}
          <div className="absolute left-0 top-4 bottom-4 w-1 bg-gradient-to-b from-transparent via-gray-300 to-transparent"></div>

          <LegendItems
            sections={sections}
            selectedSections={selectedSections}
            handleSectionSelect={handleSectionSelect}
            togglePlayingSection={togglePlayingSection}
            playingSection={playingSection}
            audio={activeTab === "Highlights" ? true : false}
            select={activeTab === "Highlights" ? true : false}
          />

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
