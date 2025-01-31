import React, { useState } from "react";
import AudioFeaturesGraph from "./AudioFeaturesGraph";
import CircularProgress from "@mui/material/CircularProgress";

const AudioFeaturesDisplay = ({
  title,
  data,
  axes,
  highlightedSections,
  color = "",
}) => {
  const [rendered, setRendered] = useState(false);
  const handleCanvasRendered = () => {
    setRendered(true);
  };

  return (
    <div className="flex flex-col items-center h-full">
      <div className="w-full h-full pb-4 z-10">
        <div className="text-center font-semibold text-slate-800">{title}</div>
        <AudioFeaturesGraph
          data={data}
          xLabels={axes.xLabels}
          yLabels={axes.yLabels}
          minY={axes.minY}
          maxY={axes.maxY}
          highlightedSections={highlightedSections}
          color={color}
          onRenderComplete={handleCanvasRendered}
        />
      </div>
    </div>
  );
};

export default AudioFeaturesDisplay;
