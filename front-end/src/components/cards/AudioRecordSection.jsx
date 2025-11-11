import { FaMicrophoneAlt } from "react-icons/fa";
import { FaFileWaveform } from "react-icons/fa6";

const AudioRecordSection = ({ audioBlob, recordingName, onRecordClick }) => {
  return (
    <div
      className="h-full flex-1 text-center px-4 py-3 flex flex-col justify-center items-center gap-1 cursor-pointer rounded-2xl transition-all duration-200"
      onClick={!audioBlob ? onRecordClick : undefined}
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
          <div className="text-lightgray font-bold">
            <span className="hover:text-lightpink">click here</span> to record
            audio
          </div>
        )}

        {audioBlob && (
          <div
            className="text-sm hover:text-lightpink transition-all duration-200 cursor-pointer"
            onClick={onRecordClick}
          >
            modify or record a different audio?
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioRecordSection;
