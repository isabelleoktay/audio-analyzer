import React, { useRef, useMemo, useEffect } from "react";
import ButtonNoOutline from "./ButtonNoOutline";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";
import { useWavesurfer } from "@wavesurfer/react";

const WaveformPlayback = ({
  file,
  playingSection,
  setPlayingSection,
  setFile,
  setAudioBuffer,
  setFeatures,
}) => {
  const waveformContainerRef = useRef(null);

  // Memoize the URL to avoid unnecessary reloads
  const fileUrl = useMemo(() => URL.createObjectURL(file), [file]);

  const { wavesurfer, isPlaying } = useWavesurfer({
    container: waveformContainerRef,
    url: fileUrl,
    waveColor: "#60a5fa",
    progressColor: "#3b82f6",
    barWidth: 2,
    cursorColor: "#3b82f6",
    height: 80,
    responsive: true,
  });

  const toggleWaveform = () => {
    if (wavesurfer) wavesurfer.playPause();

    if (wavesurfer.isPlaying) setPlayingSection("waveform");
    else setPlayingSection(null);
  };

  const removeFile = () => {
    setFile(null);
    setPlayingSection(null);
    setAudioBuffer(null);
    setFeatures(null);
  };

  useEffect(() => {
    if (playingSection !== "waveform" && wavesurfer) wavesurfer.pause();
  }, [playingSection, wavesurfer]);

  return (
    <div className="flex flex-col w-full mt-8 p-4 bg-blue-100 rounded-lg border-2 border-blue-300 border-solid shadow-md">
      <div className="flex items-center mb-1">
        <div className="text-sm font-semibold text-blue-500">{file.name}</div>
        {/* Remove File Button */}
        <ButtonNoOutline
          text="Remove File"
          handleClick={removeFile}
          fontSize="sm"
          bgColor="red-500"
          bgColorHover="red-400"
          textColor="white"
        />
      </div>
      <div className="flex items-center">
        {/* Play/Pause Button */}
        <IconButton
          aria-label={"toggle waveform playback"}
          onClick={toggleWaveform}
          className="mr-1"
        >
          {isPlaying ? (
            <PauseCircleIcon className="text-blue-500 text-3xl" />
          ) : (
            <PlayCircleIcon className="text-blue-500 text-3xl" />
          )}
        </IconButton>

        {/* Waveform Container */}
        <div className="flex-grow" ref={waveformContainerRef}></div>
      </div>
    </div>
  );
};

export default WaveformPlayback;
