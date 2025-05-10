import { useState } from "react";

const FileUploadCard = ({ onFileUpload, uploadedFile }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0]; // Get the first file only
    if (file && file.type.startsWith("audio/")) {
      onFileUpload(file); // Pass the single file to the callback
    } else {
      alert("Only audio files are allowed.");
    }
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0]; // Get the first file only
    if (file && file.type.startsWith("audio/")) {
      onFileUpload(file); // Pass the single file to the callback
    } else {
      alert("Only audio files are allowed.");
    }
  };

  return (
    <div
      className={`w-full border-dash-10-5 border rounded-3xl p-6 text-center cursor-pointer transition ${
        isDragging ? "border-electricblue bg-lightgray/20" : "border-lightgray"
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => document.getElementById("fileInput").click()}
    >
      {uploadedFile ? (
        <div>
          <p className="text-lightpink text-lg font-semibold">
            {uploadedFile.name}
          </p>
          <p className="text-sm text-lightgray cursor-pointer underline">
            choose another file
          </p>
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
