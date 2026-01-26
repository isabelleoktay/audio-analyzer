import { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.esm.js";
import IconButton from "../buttons/IconButton";
import {
  FaPlay,
  FaPause,
  FaMicrophone,
  FaStop,
  FaRedo,
  FaVolumeUp,
} from "react-icons/fa";

const SCROLLING_WAVEFORM = true;
const CONTINUOUS_WAVEFORM = false;

const AudioRecorder = ({
  maxAttempts,
  onAttemptsChange,
  analyzeMode = false,
  onAnalyze,
  showAttempts = true,
  onRecordingChange,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [attemptCount, setAttemptCount] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [recordings, setRecordings] = useState([]); // Store blobs instead of URLs
  const [selectedAttempt, setSelectedAttempt] = useState(0);

  const waveSurferRef = useRef(null);
  const recordRef = useRef(null);
  const waveformContainerRef = useRef(null);

  useEffect(() => {
    if (onRecordingChange) {
      onRecordingChange(recordings[selectedAttempt] || null);
    }
  }, [recordings, selectedAttempt, onRecordingChange]);

  // Initialize WaveSurfer with RecordPlugin
  useEffect(() => {
    if (!waveformContainerRef.current) return;

    const waveSurfer = WaveSurfer.create({
      container: waveformContainerRef.current,
      waveColor: "#E0E0E0",
      progressColor: "#FFD6E8",
      interact: true,
      height: 50,
      responsive: true,
      barWidth: 2,
      normalize: true,
    });

    const recordPlugin = RecordPlugin.create({
      renderRecordedAudio: false,
      scrollingWaveform: SCROLLING_WAVEFORM,
      continuousWaveform: CONTINUOUS_WAVEFORM,
    });

    waveSurferRef.current = waveSurfer;
    recordRef.current = waveSurfer.registerPlugin(recordPlugin);

    // Handle recording end
    recordPlugin.on("record-end", (blob) => {
      setRecordings((prev) => {
        const newRecordings = [...prev, blob];
        if (onAttemptsChange) onAttemptsChange(newRecordings.length);

        const newAttemptCount = newRecordings.length;
        setRecordedAudio(blob);
        setAttemptCount(newAttemptCount);
        setSelectedAttempt(newAttemptCount - 1);
        setIsRecording(false);
        setIsPlaying(false);

        waveSurfer.loadBlob(blob).then(() => {
          waveSurfer.seekTo(0);
          waveSurfer.toggleInteraction(true);
        });

        return newRecordings;
      });
    });

    waveSurfer.on("finish", () => {
      setIsPlaying(false);
    });

    return () => {
      waveSurfer.destroy();
    };
  }, []);

  const handleStartRecording = async () => {
    try {
      await recordRef.current.startRecording();
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
    }
  };

  const handleStopRecording = async () => {
    await recordRef.current.stopRecording();
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      waveSurferRef.current.pause();
    } else {
      waveSurferRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleRetry = () => {
    setRecordedAudio(null);
    waveSurferRef.current.empty();
    setIsPlaying(false);
    if (onAttemptsChange) onAttemptsChange(0);
  };

  const handleSelectAttempt = (index) => {
    setSelectedAttempt(index);
    setRecordedAudio(recordings[index]);
    setIsPlaying(false);

    waveSurferRef.current.loadBlob(recordings[index]).then(() => {
      waveSurferRef.current.seekTo(0);
      waveSurferRef.current.toggleInteraction(true);
    });
  };

  const isInfinite = maxAttempts === undefined || maxAttempts === null;
  const attemptsRemaining = isInfinite ? null : maxAttempts - attemptCount;
  const canRecord = isInfinite || attemptCount < maxAttempts;

  return (
    <div className="flex flex-col space-y-4">
      {/* Waveform Display with Controls */}
      <div className="flex flex-col space-y-1">
        <div className="bg-lightgray/20 p-4 rounded-3xl w-full">
          <div className="flex flex-row items-center w-full h-full">
            <div className="flex justify-center items-center">
              {!recordedAudio ? (
                isRecording ? (
                  <IconButton
                    icon={FaStop}
                    onClick={handleStopRecording}
                    colorClass="text-darkpink"
                    bgClass="bg-transparent"
                    sizeClass="w-10 h-10"
                    ariaLabel="stop recording"
                  />
                ) : (
                  <IconButton
                    icon={FaMicrophone}
                    onClick={handleStartRecording}
                    colorClass="text-lightgray"
                    bgClass="bg-transparent"
                    sizeClass="w-10 h-10"
                    ariaLabel="start recording"
                    disabled={!canRecord}
                  />
                )
              ) : (
                <IconButton
                  icon={isPlaying ? FaPause : FaPlay}
                  onClick={handlePlayPause}
                  colorClass="text-lightgray"
                  bgClass="bg-transparent"
                  sizeClass="w-10 h-10"
                  ariaLabel="play recording"
                />
              )}
            </div>
            <div className="flex-grow flex flex-col">
              <div ref={waveformContainerRef} className="w-full" />
            </div>
          </div>
        </div>

        {/* Bottom Row: Try Again Button, Attempt Selection, and Attempts Bubble */}
        <div className="flex justify-between items-center">
          {/* Try Again Button on the Left */}
          <div className={recordedAudio && canRecord ? "" : "invisible"}>
            <IconButton
              icon={FaRedo}
              onClick={handleRetry}
              colorClass="text-lightgray"
              bgClass="bg-bluegray/80"
              sizeClass="w-8 h-8"
              iconSize="w-3 h-3"
              ariaLabel="try again"
            />
          </div>

          <div className="flex flex-row gap-2">
            {/* Attempt Selection Buttons */}
            {showAttempts && recordings.length > 0 && (
              <div className="flex space-x-1">
                {recordings.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => handleSelectAttempt(index)}
                    className={`py-1 px-3 rounded-full text-xs transition-colors flex items-center justify-center gap-1 ${
                      selectedAttempt === index
                        ? "bg-electricblue text-blueblack"
                        : "bg-bluegray/80 text-lightgray hover:bg-bluegray"
                    }`}
                  >
                    <FaVolumeUp className="text-xs" />
                    <span>attempt {index + 1}</span>
                  </button>
                ))}
              </div>
            )}

            {/* Attempts Remaining Bubble on the Right */}
            {!isInfinite && (
              <div className="bg-warmyellow/50 text-lightgray px-3 py-1 rounded-full text-xs font-semibold">
                {attemptsRemaining}{" "}
                {attemptsRemaining === 1 ? "attempt" : "attempts"} left
              </div>
            )}

            {/* Analyze Button */}
            {analyzeMode && recordedAudio && (
              <button
                onClick={() => onAnalyze && onAnalyze(recordedAudio)}
                className="bg-electricblue/80 hover:bg-electricblue text-blueblack px-3 py-1 rounded-full text-xs font-bold transition-colors shadow-sm"
              >
                Analyze
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioRecorder;
