import React from "react";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import AttachmentIcon from "@mui/icons-material/Attachment";
import NoteRangeSelector from "./NoteRangeSelector/NoteRangeSelector";

const AudioUploadSection = ({
  file,
  setFile,
  minNote,
  maxNote,
  setMinNote,
  setMaxNote,
  setError,
}) => {
  const handleFile = (uploadedFile) => {
    if (uploadedFile && uploadedFile.type.startsWith("audio/")) {
      setFile(uploadedFile);
    } else {
      setError("Please upload an audio file.");
    }
  };

  const handleFileUpload = async (event) => {
    const uploadedFile = event.target.files[0];
    handleFile(uploadedFile);
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    const uploadedFile = event.dataTransfer.files[0];
    handleFile(uploadedFile);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };
  return (
    <div
      className="rounded-lg shadow-inner border-2 border-blue-200 bg-gray-50 p-6 overflow-visible w-full"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <input
        type="file"
        accept="audio/*"
        onChange={handleFileUpload}
        className="hidden"
        id="fileInput"
      />
      <label htmlFor="fileInput" className="block text-gray-600">
        <div className="flex flex-col items-center">
          <UploadFileIcon className="text-gray-600 sm:text-2xl md:text-2xl lg:text-3xl xl:text-3xl mb-1" />
          <p>
            1. Drag and drop file here or{" "}
            <span className="text-blue-500 underline font-bold cursor-pointer">
              upload file
            </span>
          </p>
        </div>
      </label>
      {file && (
        <div className="mt-1 flex items-center text-sm justify-center">
          <AttachmentIcon className="text-gray-500 mr-2" />
          <span className="text-gray-500">{file.name}</span>
        </div>
      )}
      <hr className="my-4 border-blue-300 w-full" />
      <NoteRangeSelector
        minNote={minNote}
        maxNote={maxNote}
        setMinNote={setMinNote}
        setMaxNote={setMaxNote}
      />
    </div>
  );
};

export default AudioUploadSection;
