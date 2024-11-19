import React, { useRef, useEffect, useCallback } from "react";
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
  const isMountedRef = useRef(false);

  const initializeWaveSurfer = useCallback(() => {
    if (waveSurferRef.current || !waveformContainerRef.current || !file) return;

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

    waveSurferRef.current.on("ready", () => {
      if (isMountedRef.current) {
        waveSurferRef.current.on("play", () => {
          setPlayingSection("waveform");
        });

        waveSurferRef.current.on("finish", () => {
          setPlayingSection(null);
        });
      }
    });
  }, [file, setPlayingSection]);

  const destroyWaveSurfer = useCallback(() => {
    if (waveSurferRef.current) {
      try {
        waveSurferRef.current.destroy();
      } catch (error) {
        console.error("Error during WaveSurfer destruction:", error);
      } finally {
        waveSurferRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    initializeWaveSurfer();

    return () => {
      isMountedRef.current = false;
      destroyWaveSurfer();
    };
  }, [initializeWaveSurfer, destroyWaveSurfer]);

  useEffect(() => {
    if (playingSection !== "waveform" && waveSurferRef.current) {
      waveSurferRef.current.pause();
    }
  }, [playingSection]);

  const toggleWaveform = () => {
    if (waveSurferRef.current) {
      if (waveSurferRef.current.isPlaying()) {
        waveSurferRef.current.pause();
        setPlayingSection(null);
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
    destroyWaveSurfer();
  };

  return (
    <div className="flex flex-col w-full mt-8 p-4 bg-blue-100 rounded-lg border-2 border-blue-300 border-solid">
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
