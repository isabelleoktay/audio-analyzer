import { useRef, useState, useEffect } from "react";
import { FaPlay, FaPause } from "react-icons/fa";
import SecondaryButton from "../buttons/SecondaryButton";

const AudioPlayback = ({ audioUrl }) => {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const animationFrameRef = useRef(null); // Reference for requestAnimationFrame

  useEffect(() => {
    const audio = audioRef.current;

    const updateProgress = () => {
      if (audio && !audio.paused) {
        const progressPercentage = (audio.currentTime / audio.duration) * 100;
        setProgress(progressPercentage);
        animationFrameRef.current = requestAnimationFrame(updateProgress); // Continuously update
      }
    };

    const setAudioDuration = () => {
      if (audio) {
        setDuration(audio.duration);
      }
    };

    if (audio) {
      audio.addEventListener("loadedmetadata", setAudioDuration);
      audio.addEventListener("play", () => {
        setIsPlaying(true);
        animationFrameRef.current = requestAnimationFrame(updateProgress); // Start updating progress
      });
      audio.addEventListener("pause", () => {
        setIsPlaying(false);
        cancelAnimationFrame(animationFrameRef.current); // Stop updating progress
      });
      audio.addEventListener("ended", () => {
        setIsPlaying(false);
        setProgress(0); // Reset progress when audio ends
        cancelAnimationFrame(animationFrameRef.current); // Stop updating progress
      });
    }

    return () => {
      if (audio) {
        audio.removeEventListener("loadedmetadata", setAudioDuration);
        audio.removeEventListener("play", () => setIsPlaying(true));
        audio.removeEventListener("pause", () => setIsPlaying(false));
        audio.removeEventListener("ended", () => {
          setIsPlaying(false);
          setProgress(0);
        });
      }
      cancelAnimationFrame(animationFrameRef.current); // Clean up animation frame
    };
  }, []);

  const togglePlayPause = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
  };

  const handleSeek = (e) => {
    const audio = audioRef.current;
    const newTime = (e.target.value / 100) * duration;
    if (audio) {
      audio.currentTime = newTime;
      setProgress(e.target.value);
    }
  };

  return (
    <div className="flex flex-row items-center gap-4 w-full mx-auto">
      <audio ref={audioRef} src={audioUrl} preload="metadata" />
      <SecondaryButton onClick={togglePlayPause} className="aspect-square">
        {isPlaying ? <FaPause /> : <FaPlay />}
      </SecondaryButton>
      <input
        type="range"
        min="0"
        max="100"
        value={progress}
        onChange={handleSeek}
        className="w-full"
        style={{
          appearance: "none",
          backgroundColor: "#E0E0E0",
          height: "4px",
          borderRadius: "10px",
        }}
      />
      <style>
        {`
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      background-color: #FF89BB;
      border-radius: 50%;
      cursor: pointer;
    }
    input[type="range"]::-moz-range-thumb {
      width: 16px;
      height: 16px;
      background-color: #FFD6E8;
      border-radius: 50%;
      cursor: pointer;
    }
  `}
      </style>
    </div>
  );
};

export default AudioPlayback;
