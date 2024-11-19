import React, { useRef, useEffect } from "react";
import ButtonNoOutline from "./ButtonNoOutline";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import IconButton from "@mui/material/IconButton";
import WaveSurfer from "wavesurfer.js";

const WaveformPlayback = ({
  file,
  playingSection,
  setPlayingSection,
  setFile,
  setAudioBuffer,
  setFeatures,
}) => {
  const waveSurferRef = useRef(null);
  const waveformContainerRef = useRef(null);

  const toggleWaveform = () => {
    if (waveSurferRef.current) {
      if (waveSurferRef.current.isPlaying()) {
        waveSurferRef.current.pause();
        if (playingSection === "waveform") {
          setPlayingSection(null);
        }
      } else {
        waveSurferRef.current.play();
      }
    }
  };

  const removeFile = () => {
    setFile(null);
    setPlayingSection(null);
    setAudioBuffer(null);
    setFeatures(null);
    if (waveSurferRef.current) {
      waveSurferRef.current.destroy();
      waveSurferRef.current = null;
    }
  };

  useEffect(() => {
    if (file && waveformContainerRef.current) {
      if (waveSurferRef.current) {
        waveSurferRef.current.destroy();
      }

      waveSurferRef.current = WaveSurfer.create({
        container: waveformContainerRef.current,
        waveColor: "#60a5fa",
        progressColor: "#3b82f6",
        barWidth: 2,
        cursorColor: "#3b82f6",
        height: 80,
        responsive: true,
      });

      waveSurferRef.current.load(URL.createObjectURL(file));

      waveSurferRef.current.on("play", () => {
        setPlayingSection("waveform");
      });

      waveSurferRef.current.on("finish", () => {
        setPlayingSection(null);
      });
    }

    return () => {
      if (waveSurferRef.current) {
        waveSurferRef.current.destroy();
      }
    };
  }, [file]);

  return (
    <div className="flex flex-col w-full mt-8 p-4 bg-blue-100 rounded-lg border-2 border-blue-300 border-solid">
      <div className="flex items-center mb-1">
        <div className="text-sm font-semibold text-blue-500">{file.name}</div>
        {/* Remove File Button */}
        <ButtonNoOutline
          text="Remove File"
          handleClick={removeFile}
          fontSize="xs"
          bgColor="red-500"
          bgColorHover="red-400"
          textColor="white"
        />
      </div>
      <div className="flex items-center">
        {/* Play/Pause Button */}
        <IconButton
          aria-label={playingSection === "waveform" ? "pause" : "play"}
          onClick={toggleWaveform}
          className="mr-1"
        >
          {playingSection === "waveform" ? (
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
