import { FaMicrophoneAlt } from "react-icons/fa";
import { FaFileWaveform } from "react-icons/fa6";

const AudioRecordSection = ({
  audioBlob,
  recordingName,
  onRecordClick,
  selectedAudioSource,
  onSelectAudioSource,
}) => {
  return (
    <div
      className={`h-full flex-1 text-center px-4 py-3 flex flex-col justify-center items-center gap-1 cursor-pointer rounded-2xl transition-all duration-200
     ${
       selectedAudioSource === "record"
         ? "bg-lightpink/10"
         : "hover:bg-lightpink/5"
     }`}
      onClick={() => onSelectAudioSource("record")}
    >
      {audioBlob ? (
        <FaFileWaveform className="text-2xl text-lightpink" />
      ) : (
        <FaMicrophoneAlt className="text-2xl text-lightgray" />
      )}

      <div className="flex flex-col -space-y-1">
        {audioBlob ? (
          <span className="text-lightpink font-bold">{recordingName}.wav</span>
        ) : (
          <div
            className="text-lightgray font-bold cursor-pointer hover:text-lightpink transition-all duration-200"
            onClick={onRecordClick}
          >
            click here to record audio
          </div>
        )}

        {audioBlob && (
          <div
            className="text-sm hover:text-lightpink transition-all duration-200 cursor-pointer"
            onClick={onRecordClick}
          >
            record a different audio?
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioRecordSection;
