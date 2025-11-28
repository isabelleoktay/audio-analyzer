import { FaPlay, FaPause } from "react-icons/fa";
import { FiRefreshCw } from "react-icons/fi";
import { PiRecordFill } from "react-icons/pi";
import SecondaryButton from "../buttons/SecondaryButton";
import TertiaryButton from "../buttons/TertiaryButton";

const RecordingControls = ({
  audioBlob,
  isPlaying,
  isRecording,
  recordingName,
  pulsingRecordStyle,
  onPlayPause,
  onRecordButtonClick,
  onRecordingNameChange,
  onCancelRecord,
}) => {
  return (
    <div className="flex flex-row justify-between w-full text-base">
      <div className="flex items-center space-x-2">
        {/* Play/pause button */}
        {audioBlob && (
          <SecondaryButton onClick={onPlayPause} className="aspect-square">
            {isPlaying ? (
              <FaPause className="text-base" />
            ) : (
              <FaPlay className="text-base" />
            )}
          </SecondaryButton>
        )}

        {/* Multi-function record/stop/redo button */}
        <SecondaryButton
          onClick={onRecordButtonClick}
          className="aspect-square"
        >
          {isRecording ? (
            <PiRecordFill
              className="text-base text-hotpink"
              style={pulsingRecordStyle}
            />
          ) : audioBlob ? (
            <FiRefreshCw className="text-base" />
          ) : (
            <PiRecordFill className="text-base" />
          )}
        </SecondaryButton>

        {/* Filename input */}
        <div className="relative ml-2 h-full">
          <input
            type="text"
            value={recordingName}
            onChange={(e) => onRecordingNameChange(e.target.value)}
            className="pl-2 py-1 h-full text-lightpink text-base pr-10 text-sm rounded-3xl bg-lightgray/25 focus:border-none focus:outline-none"
            placeholder="untitled"
          />
          {!recordingName.toLowerCase().endsWith(".wav") && (
            <span className="text-sm text-electricblue absolute right-2 top-1/2 -translate-y-1/2">
              .wav
            </span>
          )}
        </div>
      </div>
      <TertiaryButton onClick={onCancelRecord} className="text-base">
        back to file upload
      </TertiaryButton>
    </div>
  );
};

export default RecordingControls;
