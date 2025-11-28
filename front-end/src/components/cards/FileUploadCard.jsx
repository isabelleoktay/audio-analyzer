import { useState } from "react";

const FileUploadCard = ({ onFileUpload, inputFile, testingEnabled }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    if (!testingEnabled) {
      e.preventDefault();
      setIsDragging(true);
    }
  };

  const handleDragLeave = () => {
    if (!testingEnabled) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e) => {
    if (!testingEnabled) {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("audio/")) {
        onFileUpload(file);
      } else {
        alert("Only audio files are allowed.");
      }
    }
  };

  const handleFileInputChange = (e) => {
    if (!testingEnabled) {
      const file = e.target.files[0];
      if (file && file.type.startsWith("audio/")) {
        onFileUpload(file);
      } else {
        alert("Only audio files are allowed.");
      }
    }
  };

  return (
    <div
      className={`w-full border-dash-10-5 border rounded-3xl p-6 text-center transition ${
        isDragging ? "border-electricblue bg-lightgray/20" : "border-lightgray"
      } ${testingEnabled ? "cursor-not-allowed" : "cursor-pointer"}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => {
        if (!testingEnabled) document.getElementById("fileInput").click();
      }}
    >
      {inputFile ? (
        <div>
          <p className="text-lightpink text-lg font-semibold">
            {inputFile.name}
          </p>
          {!testingEnabled && (
            <p className="text-sm text-lightgray cursor-pointer underline">
              choose another file
            </p>
          )}
        </div>
      ) : (
        <p className="text-lightgray">
          drag and drop or{" "}
          <span className="text-electricblue font-semibold">click here</span> to
          upload your song
        </p>
      )}
      <input
        id="fileInput"
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleFileInputChange}
      />
    </div>
  );
};

export default FileUploadCard;
