import { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.esm.js";
import AudioUploadSection from "./AudioUploadSection";
import AudioRecordSection from "./AudioRecordSection";
import AudioDivider from "./AudioDivider";
import RecordingControls from "./RecordingControls";
import AudioSourceSelector from "../selectors/AudioSourceSelector";

const SCROLLING_WAVEFORM = true;
const CONTINUOUS_WAVEFORM = false;

const pulsingRecordStyle = {
  animation: "pulse 2s infinite ease-in-out",
};

const UploadAudioCard = ({ label, onAudioSourceChange, onAudioDataChange }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isRecordingMode, setIsRecordingMode] = useState(false);
  const [recordingName, setRecordingName] = useState("untitled");
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [selectedAudioSource, setSelectedAudioSource] = useState("upload"); // Default to upload

  const fileInputRef = useRef(null);
  const waveSurferRef = useRef(null);
  const recordRef = useRef(null);

  const handleSelectAudioSource = (source) => {
    setSelectedAudioSource(source);
    onAudioSourceChange?.(source);

    // Send the appropriate audio data based on selection
    if (source === "upload" && selectedFile) {
      onAudioDataChange?.({
        source: "upload",
        file: selectedFile,
        blob: null,
        url: null,
      });
    } else if (source === "record" && audioBlob) {
      onAudioDataChange?.({
        source: "record",
        file: null,
        blob: audioBlob,
        url: URL.createObjectURL(audioBlob),
        name: recordingName,
      });
    } else {
      // Clear data if switching to a source that doesn't have content
      onAudioDataChange?.({
        source: source,
        file: null,
        blob: null,
        url: null,
      });
    }
  };

  // Handle file upload
  const handleFileUpload = (files) => {
    const audioFiles = Array.from(files).filter((file) =>
      file.type.startsWith("audio/")
    );

    if (audioFiles.length > 0) {
      //   console.log("Audio file(s) selected:", audioFiles);
      setSelectedFile(audioFiles[0]);

      // Only send data if upload is the selected source
      if (selectedAudioSource === "upload") {
        onAudioDataChange?.({
          source: "upload",
          file: audioFiles[0],
          blob: null,
          url: null,
        });
      }
    } else {
      console.warn("No audio files were selected");
    }
  };

  // Click handlers
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  // Reset file selection
  const handleResetFile = () => {
    setSelectedFile(null);
    fileInputRef.current.value = "";

    // Only clear data if upload is the selected source
    if (selectedAudioSource === "upload") {
      onAudioDataChange?.({
        source: "upload",
        file: null,
        blob: null,
        url: null,
      });
    }
  };

  // Recording handlers
  const handleRecordClick = (e) => {
    e.stopPropagation();
    setIsRecordingMode(true);
  };

  const handleCancelRecord = () => {
    setIsRecordingMode(false);
  };

  const handleRecordButtonClick = async () => {
    if (isRecording) {
      await handleStopRecording();
    } else if (audioBlob) {
      handleResetRecording();
      setTimeout(() => {
        handleStartRecording();
      }, 100);
    } else {
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
    waveSurferRef.current.empty();

    // Only clear data if record is the selected source
    if (selectedAudioSource === "record") {
      onAudioDataChange?.({
        source: "record",
        file: null,
        blob: null,
        url: null,
      });
    }
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
        height: 100,
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
        setIsPlaying(false);

        // Only send data if record is the selected source
        if (selectedAudioSource === "record") {
          onAudioDataChange?.({
            source: "record",
            file: null,
            blob: blob,
            url: url,
            name: recordingName,
          });
        }

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
  }, [
    setAudioBlob,
    onAudioDataChange,
    recordingName,
    audioBlob,
    isRecordingMode,
    selectedAudioSource,
  ]);

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
        className={`w-full h-[150px] ${
          isDragging
            ? "bg-lightgray/30 border-2 border-dashed border-lightpink"
            : "bg-lightgray/15"
        } rounded-3xl p-4 text-lg transition-all duration-200 flex items-center`}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isRecordingMode ? (
          <div id="waveform" className="w-full items-center"></div>
        ) : (
          <div className="flex flex-row w-full h-full items-center justify-center text-lightgray">
            <AudioUploadSection
              selectedFile={selectedFile}
              fileInputRef={fileInputRef}
              onUploadClick={handleUploadClick}
              onResetFile={handleResetFile}
              onFileInputChange={handleFileInputChange}
            />
            <AudioDivider />
            <AudioRecordSection
              audioBlob={audioBlob}
              recordingName={recordingName}
              onRecordClick={handleRecordClick}
            />
          </div>
        )}
      </div>
      <div className="h-12 mt-1">
        {isRecordingMode ? (
          <RecordingControls
            audioBlob={audioBlob}
            isPlaying={isPlaying}
            isRecording={isRecording}
            recordingName={recordingName}
            pulsingRecordStyle={pulsingRecordStyle}
            onPlayPause={handlePlayPause}
            onRecordButtonClick={handleRecordButtonClick}
            onRecordingNameChange={setRecordingName}
            onCancelRecord={handleCancelRecord}
          />
        ) : (
          <AudioSourceSelector
            selectedSource={selectedAudioSource}
            onSourceChange={handleSelectAudioSource}
            hasUploadedFile={!!selectedFile}
            hasRecordedAudio={!!audioBlob}
          />
        )}
      </div>
    </div>
  );
};

export default UploadAudioCard;
