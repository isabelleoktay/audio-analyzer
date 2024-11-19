// import React, { useRef, useEffect } from "react";
// import ButtonNoOutline from "./ButtonNoOutline";
// import PlayCircleIcon from "@mui/icons-material/PlayCircle";
// import PauseCircleIcon from "@mui/icons-material/PauseCircle";
// import IconButton from "@mui/material/IconButton";
// import { useWavesurfer } from "@wavesurfer/react";

// const WaveformPlayback = ({
//   file,
//   playingSection,
//   setPlayingSection,
//   setFile,
//   setAudioBuffer,
//   setFeatures,
// }) => {
//   const waveSurferRef = useRef(null);
//   const waveformContainerRef = useRef(null);

//   // const toggleWaveform = () => {
//   //   if (waveSurferRef.current) {
//   //     if (waveSurferRef.current.isPlaying()) {
//   //       waveSurferRef.current.pause();
//   //       if (playingSection === "waveform") {
//   //         setPlayingSection(null);
//   //       }
//   //     } else {
//   //       waveSurferRef.current.play();
//   //     }
//   //   }
//   // };

//   const toggleWaveform = () => {
//     wavesurfer && wavesurfer.playPause();
//   };

//   const removeFile = () => {
//     setFile(null);
//     setPlayingSection(null);
//     setAudioBuffer(null);
//     setFeatures(null);
//     // if (waveSurferRef.current) {
//     //   waveSurferRef.current.destroy();
//     //   waveSurferRef.current = null;
//     // }
//   };

//   const { wavesurfer, isReady, isPlaying, currentTime } = useWavesurfer({
//     container: waveformContainerRef,
//     url: URL.createObjectURL(file),
//     waveColor: "#60a5fa",
//     progressColor: "#3b82f6",
//     barWidth: 2,
//     cursorColor: "#3b82f6",
//     height: 80,
//     responsive: true,
//   });

//   // useEffect(() => {
//   //   if (file && waveformContainerRef.current) {
//   //     if (waveSurferRef.current) {
//   //       waveSurferRef.current.destroy();
//   //     }

//   //     waveSurferRef.current = WaveSurfer.create({
//   //       container: waveformContainerRef.current,
//   //       waveColor: "#60a5fa",
//   //       progressColor: "#3b82f6",
//   //       barWidth: 2,
//   //       cursorColor: "#3b82f6",
//   //       height: 80,
//   //       responsive: true,
//   //     });

//   //     waveSurferRef.current.load(URL.createObjectURL(file));

//   //     waveSurferRef.current.on("play", () => {
//   //       setPlayingSection("waveform");
//   //     });

//   //     waveSurferRef.current.on("finish", () => {
//   //       setPlayingSection(null);
//   //     });
//   //   }

//   //   return () => {
//   //     if (waveSurferRef.current) {
//   //       waveSurferRef.current.destroy();
//   //     }
//   //   };
//   // }, [file]);

//   // useEffect(() => {
//   //   if (playingSection !== null && playingSection !== "waveform") {
//   //     waveSurferRef.current.pause();
//   //   }
//   // }, [playingSection]);

//   return (
//     <div className="flex flex-col w-full mt-8 p-4 bg-blue-100 rounded-lg border-2 border-blue-300 border-solid">
//       <div className="flex items-center mb-1">
//         <div className="text-sm font-semibold text-blue-500">{file.name}</div>
//         {/* Remove File Button */}
//         <ButtonNoOutline
//           text="Remove File"
//           handleClick={removeFile}
//           fontSize="sm"
//           bgColor="red-500"
//           bgColorHover="red-400"
//           textColor="white"
//         />
//       </div>
//       <div className="flex items-center">
//         {/* Play/Pause Button */}
//         <IconButton
//           aria-label={playingSection === "waveform" ? "pause" : "play"}
//           onClick={toggleWaveform}
//           className="mr-1"
//         >
//           {playingSection === "waveform" ? (
//             <PauseCircleIcon className="text-blue-500 text-3xl" />
//           ) : (
//             <PlayCircleIcon className="text-blue-500 text-3xl" />
//           )}
//         </IconButton>

//         {/* Waveform Container */}
//         <div className="flex-grow" ref={waveformContainerRef}></div>
//       </div>
//     </div>
//   );
// };

// export default WaveformPlayback;

import React, { useRef, useEffect, useMemo } from "react";
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

  const { wavesurfer, isReady, isPlaying, currentTime } = useWavesurfer({
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
  };

  const removeFile = () => {
    setFile(null);
    setPlayingSection(null);
    setAudioBuffer(null);
    setFeatures(null);
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
