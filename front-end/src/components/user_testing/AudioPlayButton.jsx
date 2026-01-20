import { useState, useRef, useEffect } from "react";
import { FaPlay, FaPause } from "react-icons/fa";
import IconButton from "../buttons/IconButton.jsx";

const AudioPlayButton = ({ audioUrl }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  useEffect(() => {
    audioRef.current = new Audio(audioUrl);
    audioRef.current.onended = () => setIsPlaying(false);

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, [audioUrl]);

  const togglePlay = () => {
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <IconButton
      icon={isPlaying ? FaPause : FaPlay}
      onClick={togglePlay}
      colorClass="text-lightgray"
      bgClass="bg-bluegray/80"
      sizeClass="w-12 h-12"
      iconSize="w-4 h-4"
      ariaLabel={isPlaying ? "pause reference" : "play reference"}
    />
  );
};

export default AudioPlayButton;
