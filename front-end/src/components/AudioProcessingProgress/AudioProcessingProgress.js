import React from "react";
import CircularProgress from "@mui/material/CircularProgress";
import LinearProgress from "@mui/material/LinearProgress";

const AudioProcessingProgress = ({ statusMessage, progress }) => {
  return (
    <div className="w-full">
      {statusMessage && (
        <div className="text-blue-500 font-extrabold text-xl mb-4 wave-animation">
          {statusMessage}
        </div>
      )}
      {/* <CircularProgress size={50} /> */}
      <LinearProgress variant="determinate" value={progress} />
    </div>
  );
};

export default AudioProcessingProgress;
