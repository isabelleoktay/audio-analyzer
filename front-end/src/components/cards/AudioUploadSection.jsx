import { FiUpload, FiFile } from "react-icons/fi";
import { IoMdCloseCircle } from "react-icons/io";

const AudioUploadSection = ({
  selectedFile,
  fileInputRef,
  onUploadClick,
  onResetFile,
  onFileInputChange,
}) => {
  return (
    <div
      className="h-full flex-1 text-center px-4 py-3 flex flex-col justify-center items-center gap-1 cursor-pointer rounded-2xl transition-all duration-200"
      onClick={!selectedFile ? onUploadClick : undefined} // Only clickable when no file
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={onFileInputChange}
        accept="audio/*"
        className="hidden"
      />

      {selectedFile ? (
        <FiFile className="text-2xl text-lightpink" />
      ) : (
        <FiUpload className="text-2xl text-lightgray" />
      )}

      <div className="flex flex-col -space-y-1 text-lg">
        <div className="text-lightgray">
          {selectedFile ? (
            <div className="flex flex-row items-center justify-center text-lightpink gap-1">
              <span className="font-bold">{selectedFile.name}</span>
              <IoMdCloseCircle
                onClick={(e) => {
                  e.stopPropagation(); // Prevent triggering parent onClick
                  onResetFile();
                }}
                className="self-center flex-shrink-0 text-lightgray hover:text-lightpink transition-all duration-200"
              />
            </div>
          ) : (
            <div className="font-bold">
              drag and drop or{" "}
              <span className="hover:text-lightpink">click here</span> to upload
              audio
            </div>
          )}
        </div>
        {selectedFile && (
          <div
            className="text-sm hover:text-lightpink transition-all duration-200 hover:cursor-pointer"
            onClick={(e) => {
              e.stopPropagation(); // Prevent triggering parent onClick
              onUploadClick();
            }}
          >
            upload a different audio file?
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioUploadSection;
