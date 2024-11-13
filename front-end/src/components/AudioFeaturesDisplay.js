import React from "react";
import AudioFeaturesGraph from "./AudioFeaturesGraph";

const AudioFeaturesDisplay = ({ title, data, axes, highlightedSections }) => {
  return (
    <div className="flex flex-col items-center h-full">
      <div className="text-center font-semibold text-slate-800">{title}</div>
      <AudioFeaturesGraph
        data={data}
        xLabels={axes.xLabels}
        yLabels={axes.yLabels}
        minY={axes.minY}
        maxY={axes.maxY}
        highlightedSections={highlightedSections}
      />
    </div>
  );
};

export default AudioFeaturesDisplay;
