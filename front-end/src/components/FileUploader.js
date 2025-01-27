import React, { useCallback, useState, useEffect } from "react";
import CircularProgress from "@mui/material/CircularProgress";
import { processAudio, uploadAudio } from "../utils/api";
import ButtonNoOutline from "./ButtonNoOutline";
import AudioUploadContainer from "./AudioUploadContainer/AudioUploadContainer";

const FileUploader = ({
  audioContextRef,
  setFile,
  setFile2,
  file,
  file2,
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
  const [analysisType, setAnalysisType] = useState(null);

  const noteRegex = /^[A-Ga-g]#?[0-9]$/;

  const validateNotes = () => {
    return noteRegex.test(minNote) && noteRegex.test(maxNote);
  };

  const isFormValid = file && validateNotes();

  const getAudioBuffer = useCallback(
    async (audioFile) => {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext ||
          window.webkitAudioContext)();
      }
      const reader = new FileReader();
      return new Promise((resolve, reject) => {
        reader.onload = (event) => {
          audioContextRef.current.decodeAudioData(
            event.target.result,
            (buffer) => {
              resolve(buffer);
            },
            reject
          );
        };
        reader.readAsArrayBuffer(audioFile);
      });
    },
    [audioContextRef]
  );

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

  // eslint-disable-next-line
  const uploadAudioBuffer = async (file) => {
    try {
      await uploadAudio(file);
    } catch (error) {
      console.error("Error uploading audio:", error);
    }
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

  useEffect(() => {
    if (file) {
      setError("");
    }
  }, [file]);

  useEffect(() => {
    if (!file) {
      setError("Please upload a file.");
    } else if (!noteRegex.test(minNote) || !noteRegex.test(maxNote)) {
      setError("Please enter valid min and max notes (e.g., C4, G#3).");
    } else {
      setError("");
    }
  }, [isFormValid, file, minNote, maxNote]);

  return (
    <div
      // onDrop={handleDrop}
      // onDragOver={handleDragOver}
      className="mt-8 p-8 border-2 shadow-md border-dashed border-blue-300 bg-blue-100 text-center rounded-lg w-full flex flex-col items-center justify-center"
    >
      {!analysisType ? (
        <div className="w-full items-center flex flex-col justify-center">
          <p className="text-gray-600 text-lg mb-4">
            Welcome to the audio analyzer! Please choose your analysis method:
          </p>
          <div className="flex flex-row space-x-4">
            <ButtonNoOutline
              text="Analyze Single Audio"
              fontSize="lg"
              bgColor="blue-500"
              bgColorHover="blue-400"
              handleClick={() => setAnalysisType("singleAudio")}
            />
            <ButtonNoOutline
              text="Compare Two Audios"
              fontSize="lg"
              bgColor="blue-500"
              bgColorHover="blue-400"
              handleClick={() => setAnalysisType("doubleAudio")}
              disabled={true}
            />
          </div>
        </div>
      ) : (
        <div className="w-1/2">
          {loading ? (
            <div className="w-full">
              <p className="text-gray-600 text-lg mb-4">Processing audio</p>
              <CircularProgress size={50} />
            </div>
          ) : (
            <div className="w-full">
              <AudioUploadContainer
                analysisType={analysisType}
                file={file}
                setFile={setFile}
                file2={file2}
                setFile2={setFile2}
                minNote={minNote}
                maxNote={maxNote}
                setMinNote={setMinNote}
                setMaxNote={setMaxNote}
                setError={setError}
              />
              {/* <div
                className={`audio-upload-section ${
                  analysisType ? "show" : ""
                } flex flex-row space-x-12`}
              >
                <div>
                  {analysisType === "doubleAudio" && (
                    <div className="text-xl font-bold text-gray-600 p-2">
                      Audio 1
                    </div>
                  )}
                  <AudioUploadSection
                    file={file}
                    setFile={setFile}
                    setError={setError}
                    minNote={minNote}
                    maxNote={maxNote}
                    setMinNote={setMinNote}
                    setMaxNote={setMaxNote}
                  />
                </div>
                {analysisType === "doubleAudio" && (
                  <div>
                    <div className="text-xl font-bold text-gray-600 p-2">
                      Audio 2
                    </div>

                    <AudioUploadSection
                      file={file2}
                      setFile={setFile2}
                      setError={setError}
                      minNote={minNote}
                      maxNote={maxNote}
                      setMinNote={setMinNote}
                      setMaxNote={setMaxNote}
                    />
                  </div>
                )}
              </div> */}
              <div className="mt-4 mb-2">
                <ButtonNoOutline
                  text={
                    analysisType === "doubleAudio"
                      ? "Compare Both Audios"
                      : "Process Audio"
                  }
                  handleClick={handleSubmit}
                  fontSize="base"
                  bgColor="blue-500"
                  bgColorHover="blue-400"
                  textColor="white"
                  textColorHover="white"
                  disabled={!isFormValid}
                  width="w-full"
                />
              </div>
              {error && (
                <div
                  aria-live="assertive"
                  className={`w-full mt-2 text-red-700 text-sm font-semibold bg-red-100 border border-red-400 rounded py-1 px-3 whitespace-normal break-words`}
                >
                  {error}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FileUploader;
