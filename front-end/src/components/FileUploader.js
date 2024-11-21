import React, { useCallback, useState } from "react";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import AttachmentIcon from "@mui/icons-material/Attachment";
import CircularProgress from "@mui/material/CircularProgress";
import { processAudio, uploadAudio } from "../utils/api";
import ButtonNoOutline from "./ButtonNoOutline";

const FileUploader = ({
  audioContext,
  setFile,
  file,
  setAudioBuffer,
  setAudioData,
  setFeatures,
  minNote,
  maxNote,
  setMinNote,
  setMaxNote,
}) => {
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

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

  const uploadAudioBuffer = async (file) => {
    try {
      await uploadAudio(file);
    } catch (error) {
      console.error("Error uploading audio:", error);
    }
  };

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

  const handleMinNoteChange = (event) => {
    setMinNote(event.target.value);
  };

  const handleMaxNoteChange = (event) => {
    setMaxNote(event.target.value);
  };

  const handleSubmit = async () => {
    if (!validateNotes()) {
      return;
    }
    if (!file) {
      setError("Please upload an audio file.");
      return;
    }
    setLoading(true);
    const buffer = await getAudioBuffer(file);
    setAudioBuffer(buffer);
    setAudioData(buffer.getChannelData(0));

    await processAudioBuffer(file);
    // await uploadAudioBuffer(file);
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      className="mt-8 p-8 border-2 border-dashed border-blue-300 bg-blue-100 text-center rounded-lg w-full cursor-default"
    >
      {loading ? (
        <div>
          <p className="text-gray-600 text-lg mb-4">Processing audio</p>
          <CircularProgress size={50} />
        </div>
      ) : (
        <div>
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
          {file && (
            <div className="mt-1 flex items-center text-sm justify-center">
              <AttachmentIcon className="text-gray-500 mr-2" />
              <span className="text-gray-500">{file.name}</span>
            </div>
          )}
          <div className="my-4">
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
              <p
                className={`transition-opacity duration-500 ease-in-out ${
                  error ? "opacity-90" : "opacity-0"
                } text-white rounded font-semibold bg-red-500 py-1 px-2 mt-4`}
              >
                {error}
              </p>
            )}
          </div>
          <ButtonNoOutline
            text="Process Audio"
            handleClick={handleSubmit}
            fontSize="base"
            bgColor="blue-500"
            bgColorHover="blue-400"
            textColor="white"
            textColorHover="white"
          />
        </div>
      )}
    </div>
  );
};

export default FileUploader;
