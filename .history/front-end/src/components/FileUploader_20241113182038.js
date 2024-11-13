import React from "react";

const FileUploader = ({
  audioContext,
  setFile,
  setAudioBuffer,
  setAudioData,
  setFeatures,
}) => {
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
      const processAudioResult = await processAudio(file);
      setFeatures(processAudioResult);

      console.log("PYTHON RESULT");
      console.log(processAudioResult);
    } catch (error) {
      console.error("Error processing audio:", error);
    }
  };

  const handleFile = async (uploadedFile) => {
    if (uploadedFile && uploadedFile.type.startsWith("audio/")) {
      setFile(uploadedFile);
      const buffer = await getAudioBuffer(uploadedFile);
      setAudioBuffer(buffer);
      setAudioData(buffer.getChannelData(0));
      await processAudioBuffer(uploadedFile);
    } else {
      alert("Please upload an audio file.");
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

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      className="mt-8 p-8 border-2 border-dashed border-blue-300 bg-blue-100 text-center rounded-lg w-3/4 max-w-lg cursor-default"
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
    </div>
  );
};
