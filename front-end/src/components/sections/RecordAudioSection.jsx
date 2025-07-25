import { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.esm.js";
import SecondaryButton from "../buttons/SecondaryButton";
import TertiaryButton from "../buttons/TertiaryButton";

const SCROLLING_WAVEFORM = true;
const CONTINUOUS_WAVEFORM = false;

const RecordAudioSection = ({
  setUploadedFile,
  setInRecordMode,
  audioBlob,
  setAudioBlob,
  audioName,
  setAudioName,
  setAudioURL,
  handleDownloadRecording,
  testingEnabled,
  feedbackStage,
  onSubmitRecording,
  onChangeAttemptCount,
  updateSubjectData,
  attemptCount = 0,
  className = "",
  testGroup,
  setIsProceedButtonEnabled,
  isProceedButtonEnabled,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const waveSurferRef = useRef(null);
  const recordRef = useRef(null);

  const handleStartRecording = async () => {
    if (testingEnabled) {
      onChangeAttemptCount();
    }
    await recordRef.current.startRecording();
    setIsRecording(true);
  };

  const handleStopRecording = async () => {
    await recordRef.current.stopRecording();
    setIsRecording(false);
    if (
      testingEnabled &&
      testGroup === "none" &&
      feedbackStage === "during" &&
      !isProceedButtonEnabled
    ) {
      setIsProceedButtonEnabled(true);
    }
  };

  const handleResetRecording = () => {
    if (testingEnabled) {
      updateSubjectData();
      if (setUploadedFile) setUploadedFile(null);
    }
    setAudioBlob(null);
    setAudioURL(null);
    if (!testingEnabled) setAudioName("untitled.wav");
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

  const handleRename = (e) => {
    setAudioName(e.target.value);
  };

  const handleAnalyze = () => {
    if (testingEnabled) {
      onSubmitRecording();
    } else {
      if (audioBlob) {
        const file = new File([audioBlob], audioName, { type: "audio/wav" });
        setUploadedFile(file);
      }
      setInRecordMode(false);
    }
  };

  useEffect(() => {
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
  }, [setUploadedFile, setAudioBlob, setAudioURL, audioBlob]);

  return (
    <div className={`flex flex-col w-full ${className}`}>
      <div className="flex flex-row justify-between items-center align-bottom mb-2 w-full">
        <div>
          {(audioBlob || testingEnabled) && (
            <input
              type="text"
              value={audioName}
              onChange={handleRename}
              disabled={testingEnabled}
              className="w-full overflow-hidden text-ellipsis whitespace-nowrap text-lightgray text-xl font-semibold bg-transparent focus:outline-none"
            />
          )}
        </div>
        {!testingEnabled && (
          <div
            onClick={handleAnalyze}
            className="text-sm text-lightgray opacity-75 hover:opacity-100 cursor-pointer"
          >
            return to file upload
          </div>
        )}
      </div>
      <div
        id="waveform"
        className="w-full bg-lightgray/25 rounded-3xl p-6 h-full mb-2"
      ></div>
      {audioBlob ? (
        <div className="flex flex-row justify-between">
          <div className="flex items-center space-x-2">
            <SecondaryButton onClick={handlePlayPause}>
              {isPlaying ? "pause" : "play"}
            </SecondaryButton>
            {!(feedbackStage === "during" && testGroup === "none") &&
              attemptCount < 3 && (
                <SecondaryButton onClick={handleResetRecording}>
                  redo
                </SecondaryButton>
              )}
            {!testingEnabled && (
              <SecondaryButton onClick={handleDownloadRecording}>
                download
              </SecondaryButton>
            )}
          </div>
          {testingEnabled === false ? (
            <TertiaryButton onClick={handleAnalyze}>
              submit recording
            </TertiaryButton>
          ) : feedbackStage === "during" && testGroup === "none" ? (
            <TertiaryButton onClick={handleAnalyze}>
              make another recording
            </TertiaryButton>
          ) : (
            <TertiaryButton onClick={handleAnalyze}>
              {feedbackStage !== "during" ? "submit recording" : "analyze"}
            </TertiaryButton>
          )}
        </div>
      ) : (
        <div className="flex items-center space-x-2">
          {isRecording ? (
            <SecondaryButton onClick={handleStopRecording}>
              stop
            </SecondaryButton>
          ) : (
            <SecondaryButton onClick={handleStartRecording}>
              record
            </SecondaryButton>
          )}
        </div>
      )}
    </div>
  );
};

export default RecordAudioSection;
