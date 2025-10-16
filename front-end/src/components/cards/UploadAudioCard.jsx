import { useState, useRef, useEffect } from "react";
import { FiUpload, FiFile, FiRefreshCw } from "react-icons/fi";
import { FaMicrophoneAlt, FaPlay, FaPause } from "react-icons/fa";
import { PiRecordFill } from "react-icons/pi";
import SecondaryButton from "../buttons/SecondaryButton";
import TertiaryButton from "../buttons/TertiaryButton";
import WaveSurfer from "wavesurfer.js";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.esm.js";

const SCROLLING_WAVEFORM = true;
const CONTINUOUS_WAVEFORM = false;

const pulsingRecordStyle = {
  animation: "pulse 2s infinite ease-in-out",
};

const UploadAudioCard = ({ label }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isRecordingMode, setIsRecordingMode] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioURL, setAudioURL] = useState(null);
  const fileInputRef = useRef(null);
  const waveSurferRef = useRef(null);
  const recordRef = useRef(null);

  // Handle file upload
  const handleFileUpload = (files) => {
    // Filter to accept only audio files
    const audioFiles = Array.from(files).filter((file) =>
      file.type.startsWith("audio/")
    );

    if (audioFiles.length > 0) {
      console.log("Audio file(s) selected:", audioFiles);
      setSelectedFile(audioFiles[0]); // Store the first selected audio file
      // Add your file handling logic here
    } else {
      console.warn("No audio files were selected");
      // Optional: Show error message to user
    }
  };

  // Click handlers
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  // Reset file selection
  const handleResetFile = () => {
    setSelectedFile(null);
    fileInputRef.current.value = ""; // Clear the file input
  };

  // Recording handlers
  const handleRecordClick = () => {
    setIsRecordingMode(true);
  };

  const handleCancelRecord = () => {
    setIsRecordingMode(false);
  };

  const handleRecordButtonClick = async () => {
    if (isRecording) {
      // If currently recording, stop the recording
      await handleStopRecording();
    } else if (audioBlob) {
      // If we have a previous recording, reset and start new recording
      handleResetRecording();
      setTimeout(() => {
        handleStartRecording();
      }, 100);
    } else {
      // If no recording exists, start recording
      handleStartRecording();
    }
  };

  const handleStartRecording = async () => {
    await recordRef.current.startRecording();
    setIsRecording(true);
  };

  const handleStopRecording = async () => {
    await recordRef.current.stopRecording();
    setIsRecording(false);
  };

  const handleResetRecording = () => {
    setAudioBlob(null);
    setAudioURL(null);
    waveSurferRef.current.empty();
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      waveSurferRef.current.pause();
    } else {
      waveSurferRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Drag handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    handleFileUpload(files);
  };

  const handleFileInputChange = (e) => {
    const files = e.target.files;
    handleFileUpload(files);
  };

  useEffect(() => {
    if (isRecordingMode) {
      const waveSurfer = WaveSurfer.create({
        container: "#waveform",
        waveColor: "rgb(255, 214, 232)",
        progressColor: "rgb(255, 137, 187)",
        interact: true,
      });

      const recordPlugin = RecordPlugin.create({
        renderRecordedAudio: false,
        scrollingWaveform: SCROLLING_WAVEFORM,
        continuousWaveform: CONTINUOUS_WAVEFORM,
      });

      waveSurferRef.current = waveSurfer;
      recordRef.current = waveSurfer.registerPlugin(recordPlugin);

      if (audioBlob) {
        waveSurfer.loadBlob(audioBlob).then(() => {
          waveSurfer.seekTo(0);
          waveSurfer.toggleInteraction(true);
        });
      }

      recordPlugin.on("record-end", (blob) => {
        const url = URL.createObjectURL(blob);
        setAudioBlob(blob);
        setAudioURL(url);
        setIsPlaying(false);
        waveSurfer.loadBlob(blob).then(() => {
          waveSurfer.seekTo(0);
          waveSurfer.toggleInteraction(true);
        });
      });

      waveSurfer.on("finish", () => {
        setIsPlaying(false);
      });

      return () => {
        waveSurfer.destroy();
      };
    }
  }, [setAudioBlob, setAudioURL, audioBlob, isRecordingMode]);

  useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = `
      @keyframes pulse {
        0%, 100% { color: rgb(255, 137, 187); }
        50% { color: #ff1493; } /* hot pink */
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  return (
    <div className="w-full flex flex-col gap-1">
      <div className="text-2xl font-medium text-lightpink tracking-wide">
        {label}
      </div>
      <div
        className={`w-full min-h-[200px] ${
          isDragging
            ? "bg-lightgray/40 border-2 border-dashed border-lightpink"
            : "bg-lightgray/25"
        } rounded-3xl p-6 text-lg transition-all duration-200 flex items-center`}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isRecordingMode ? (
          /* Recording Mode Layout */
          <div id="waveform" className="w-full h-full"></div>
        ) : (
          /* Normal Upload/Record Choice Layout */
          <div className="flex flex-row w-full items-center justify-center text-lightgray">
            {/* Hidden file input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept="audio/*"
              className="hidden"
            />

            {/* Upload section or File info */}
            <div className="flex-1 text-center px-4 py-3 flex flex-col items-center gap-1">
              {selectedFile ? (
                <FiFile className="text-2xl text-lightpink" />
              ) : (
                <FiUpload className="text-2xl text-lightgray" />
              )}
              <div className="flex flex-col -space-y-1">
                <div className="text-lightgray font-bold">
                  {selectedFile ? (
                    <span className="text-lightpink">{selectedFile.name}</span>
                  ) : (
                    <>
                      drag and drop or{" "}
                      <span
                        className="cursor-pointer hover:underline"
                        onClick={handleUploadClick}
                      >
                        click here to upload
                      </span>
                    </>
                  )}
                </div>
                {selectedFile ? (
                  <div
                    className="text-sm hover:underline hover:cursor-pointer"
                    onClick={handleResetFile}
                  >
                    upload a different audio file?
                  </div>
                ) : (
                  <div className="text-sm">a reference audio</div>
                )}
              </div>
            </div>

            <div className="flex items-center px-4">
              <div className="h-px w-12 bg-lightgray"></div>
              <div className="px-3 font-medium text-xl text-lightgray">OR</div>
              <div className="h-px w-12 bg-lightgray"></div>
            </div>

            <div className="flex-1 text-center px-4 flex flex-col items-center gap-1">
              <FaMicrophoneAlt
                className="text-2xl text-lightgray cursor-pointer"
                onClick={handleRecordClick}
              />
              <div className="flex flex-col -space-y-1">
                <div
                  className="text-lightgray font-bold cursor-pointer hover:underline"
                  onClick={handleRecordClick}
                >
                  click here to record
                </div>
                <div className="text-sm">your own reference audio</div>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="h-12 mt-1">
        {" "}
        {/* Fixed height container */}
        {isRecordingMode && (
          <div className="flex flex-row justify-between w-full">
            <div className="flex items-center space-x-2">
              {/* Play/pause button */}
              {audioBlob && (
                <SecondaryButton
                  onClick={handlePlayPause}
                  className="aspect-square"
                >
                  {isPlaying ? (
                    <FaPause className="text-lg" />
                  ) : (
                    <FaPlay className="text-lg" />
                  )}
                </SecondaryButton>
              )}

              {/* Multi-function record/stop/redo button */}
              <SecondaryButton
                onClick={handleRecordButtonClick}
                className="aspect-square"
              >
                {isRecording ? (
                  <PiRecordFill
                    className="text-lg text-hotpink"
                    style={pulsingRecordStyle}
                  />
                ) : audioBlob ? (
                  <FiRefreshCw className="text-lg" />
                ) : (
                  <PiRecordFill className="text-lg" />
                )}
              </SecondaryButton>
            </div>
            <TertiaryButton onClick={handleCancelRecord}>
              back to file upload
            </TertiaryButton>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadAudioCard;
