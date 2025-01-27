import React, { useRef, useEffect } from "react";
import AudioUploadSection from "../AudioUploadSection";
import "./AudioUploadContainer.css";

const AudioUploadContainer = ({
  analysisType,
  file,
  setFile,
  file2,
  setFile2,
  minNote,
  maxNote,
  setMinNote,
  setMaxNote,
  setError,
}) => {
  const sectionRef = useRef(null);

  useEffect(() => {
    const section = sectionRef.current;
    if (analysisType) {
      // Expand: Set to content height
      section.style.maxHeight = `${section.scrollHeight}px`;
      section.style.opacity = 1;
    } else {
      // Collapse: Reset maxHeight
      section.style.maxHeight = 0;
      section.style.opacity = 0;
    }
  }, [analysisType]);

  return (
    <div
      ref={sectionRef}
      className={`audio-upload-section flex flex-row space-x-12 min-w-0 overflow-visible`}
    >
      <div className="flex-1">
        {analysisType === "doubleAudio" && (
          <div className="text-xl font-bold text-gray-600 p-2 w-full">
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
      {/* {analysisType === "doubleAudio" && (
        <div>
          <div className="text-xl font-bold text-gray-600 p-2">Audio 2</div>
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
      )} */}
    </div>
  );
};

export default AudioUploadContainer;
