import React, { useCallback, useState } from "react";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { processAudio } from "../utils/api";

const FileUploader = ({
  audioContext,
  setFile,
  setAudioBuffer,
  setAudioData,
  setFeatures,
  minNote,
  maxNote,
  setMinNote,
  setMaxNote,
}) => {
  const [error, setError] = useState("");

  const noteRegex = /^[A-Ga-g]#?[0-9]$/;

  const validateNotes = () => {
    if (!noteRegex.test(minNote) || !noteRegex.test(maxNote)) {
      setError(
        "Invalid note format. Please use a format like C4, F#3, G3, etc."
      );
      return false;
    }
    setError("");
    return true;
  };

  const getAudioBuffer = useCallback(async (audioFile) => {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    const reader = new FileReader();
    return new Promise((resolve, reject) => {
      reader.onload = (event) => {
        audioContext.decodeAudioData(
          event.target.result,
          (buffer) => {
            resolve(buffer);
          },
          reject
        );
      };
      reader.readAsArrayBuffer(audioFile);
    });
  }, []);

  const processAudioBuffer = async (file) => {
    try {
      const processAudioResult = await processAudio(file, minNote, maxNote);
      setFeatures(processAudioResult);

      // console.log("PYTHON RESULT");
      // console.log(processAudioResult);
    } catch (error) {
      console.error("Error processing audio:", error);
      setError("Error processing audio. Please try again.");
    }
  };

  const handleFile = async (uploadedFile) => {
    if (uploadedFile && uploadedFile.type.startsWith("audio/")) {
      if (!validateNotes()) {
        return;
      }
      setFile(uploadedFile);
      const buffer = await getAudioBuffer(uploadedFile);
      setAudioBuffer(buffer);
      setAudioData(buffer.getChannelData(0));
      await processAudioBuffer(uploadedFile);
    } else {
      setError("Please upload an audio file.");
    }
  };

  const handleFileUpload = async (event) => {
    const uploadedFile = event.target.files[0];
    await handleFile(uploadedFile);
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    const uploadedFile = event.dataTransfer.files[0];
    await handleFile(uploadedFile);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleMinNoteChange = (event) => {
    setMinNote(event.target.value);
  };

  const handleMaxNoteChange = (event) => {
    setMaxNote(event.target.value);
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      className="mt-8 p-8 border-2 border-dashed border-blue-300 bg-blue-100 text-center rounded-lg w-full cursor-default"
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
            Drag and drop file here or{" "}
            <span className="text-blue-500 underline font-bold cursor-pointer">
              upload file
            </span>
          </p>
        </div>
      </label>
      <div className="mt-4">
        <label className="block text-gray-600">
          Min Note:
          <input
            type="text"
            value={minNote}
            onChange={handleMinNoteChange}
            className="ml-2 p-1 border rounded"
            placeholder="e.g., F3"
          />
        </label>
        <label className="block text-gray-600 mt-2">
          Max Note:
          <input
            type="text"
            value={maxNote}
            onChange={handleMaxNoteChange}
            className="ml-2 p-1 border rounded"
            placeholder="e.g., B6"
          />
        </label>
        {error && (
          <p className="text-white rounded font-semibold bg-red-500 opacity-90 py-1 px-2 mt-4">
            {error}
          </p>
        )}
      </div>
    </div>
  );
};

export default FileUploader;
