import { startTransition } from "react";
import { v4 as uuidv4 } from "uuid";

import InstrumentSelectionCards from "../components/cards/InstrumentSelectionCards.jsx";
import InstrumentButton from "../components/buttons/InstrumentButton.jsx";
import RecordAudioSection from "../components/sections/RecordAudioSection.jsx";
import FileUploadSection from "../components/sections/FileUploadSection.jsx";
import AnalysisButtons from "../components/buttons/AnalysisButtons.jsx";
import GraphWithWaveform from "../components/visualizations/GraphWithWaveform.jsx";
import TertiaryButton from "../components/buttons/TertiaryButton.jsx";
import Tooltip from "../components/text/Tooltip.jsx";
import { instrumentButtons } from "../config/instrumentButtons.js";

/**
 * The `Analyzer` component is the main page for analyzing audio files.
 * It allows users to select an instrument, upload or record audio, and analyze features such as pitch or dynamics.
 *
 * @component
 * @param {Object} props - The props passed to the component.
 * @param {string} props.selectedInstrument - The currently selected instrument.
 * @param {Function} props.setSelectedInstrument - Function to update the selected instrument.
 * @param {File} props.uploadedFile - The uploaded audio file.
 * @param {Function} props.setUploadedFile - Function to update the uploaded file.
 * @param {boolean} props.inRecordMode - Whether a user is recording audio to be uploaded.
 * @param {Function} props.setInRecordMode - Function to toggle recording mode.
 * @param {Blob} props.audioBlob - The audio blob data for recording.
 * @param {Function} props.setAudioBlob - Function to update the audio blob.
 * @param {string} props.audioName - The name of the audio file.
 * @param {Function} props.setAudioName - Function to update the audio file name.
 * @param {string} props.audioURL - The URL of the uploaded or recorded audio.
 * @param {Function} props.setAudioURL - Function to update the audio URL.
 * @param {string} props.selectedAnalysisFeature - The selected audio analysis feature.
 * @param {Function} props.setSelectedAnalysisFeature - Function to update the selected analysis feature.
 * @param {Object} props.audioFeatures - Extracted features from the audio.
 * @param {Function} props.setAudioFeatures - Function to update the audio features.
 * @param {string} props.audioUuid - A unique identifier for the audio session.
 * @param {Function} props.setAudioUuid - Function to update the audio UUID.
 * @param {boolean} props.uploadsEnabled - Whether uploads are enabled.
 * @param {string} props.tooltipMode - The mode for displaying tooltips.
 * @returns {JSX.Element} The rendered `Analyzer` component.
 */
const Analyzer = ({
  selectedInstrument,
  setSelectedInstrument,
  uploadedFile,
  setUploadedFile,
  inRecordMode,
  setInRecordMode,
  audioBlob,
  setAudioBlob,
  audioName,
  setAudioName,
  audioURL,
  setAudioURL,
  selectedAnalysisFeature,
  setSelectedAnalysisFeature,
  audioFeatures,
  setAudioFeatures,
  audioUuid,
  setAudioUuid,
  uploadsEnabled,
  tooltipMode,
}) => {
  // Handles the selection of an instrument.
  // Resets audio-related state if audio features are already present.
  const handleInstrumentSelect = (instrument) => {
    startTransition(() => {
      setSelectedInstrument(instrument);
      if (Object.keys(audioFeatures).length > 0) {
        console.log("Resetting audio features");
        setUploadedFile(null);
        setAudioBlob(null);
        setAudioName("untitled.wav");
        setAudioURL(null);
        setAudioFeatures({});
        setSelectedAnalysisFeature(null);
        setAudioUuid(() => uuidv4());
      }
    });
  };

  // Handles the upload of an audio file.
  // Updates the uploaded file and generates a URL for it.
  const handleFileUpload = (file) => {
    setUploadedFile(file);
    setAudioURL(URL.createObjectURL(file));
  };

  // Switches the app to recording mode.
  const handleSwitchToRecordMode = (e) => {
    e.stopPropagation();
    setInRecordMode(true);
  };

  // Handles the selection of an audio analysis feature.
  const handleAnalysisFeatureSelect = (feature) => {
    setSelectedAnalysisFeature(feature);
  };

  // Downloads the currently recorded or uploaded audio file.
  const handleDownloadRecording = () => {
    const link = document.createElement("a");
    link.href = audioURL;
    link.download = audioName;
    link.click();
  };

  // Resets the audio-related state when the user changes the file.
  const handleChangeFile = () => {
    setSelectedAnalysisFeature(null);
    setUploadedFile(null);
    setAudioBlob(null);

    const newAudioName = "untitled.wav";
    setAudioName(newAudioName);

    setAudioURL(null);
    setAudioFeatures({});
    setInRecordMode(false);
    setAudioUuid(() => uuidv4());
  };

  return (
    <div className="flex flex-col w-full items-center min-h-[calc(100vh-4rem)]">
      {selectedInstrument ? (
        /* Show smaller intrument selection buttons at the top of screen if an instrument is already selected. */
        <div className="mt-28">
          <Tooltip
            text="select an instrument to analyze"
            show={tooltipMode === "global"}
            tooltipMode={tooltipMode}
            className="w-full"
          >
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-16">
              {instrumentButtons.map((inst) => (
                <InstrumentButton
                  key={inst.label}
                  onClick={() => handleInstrumentSelect(inst.label)}
                  selected={selectedInstrument === inst.label}
                >
                  {inst.label}
                </InstrumentButton>
              ))}
            </div>
          </Tooltip>

          {/* Toggle between the recording and upload interface */}
          <div className="flex flex-col items-center justify-center w-full">
            {inRecordMode ? (
              <RecordAudioSection
                setUploadedFile={setUploadedFile}
                setInRecordMode={setInRecordMode}
                audioBlob={audioBlob}
                setAudioBlob={setAudioBlob}
                audioName={audioName}
                setAudioName={setAudioName}
                audioURL={audioURL}
                setAudioURL={setAudioURL}
                handleDownloadRecording={handleDownloadRecording}
              />
            ) : (
              !selectedAnalysisFeature && (
                <Tooltip
                  text="upload an audio file (monophonic for violin or voice)"
                  show={tooltipMode === "global"}
                  tooltipMode={tooltipMode}
                  className="w-full"
                >
                  <FileUploadSection
                    handleSwitchToRecordMode={handleSwitchToRecordMode}
                    handleFileUpload={handleFileUpload}
                    uploadedFile={uploadedFile}
                    className="mb-8 w-full"
                  />
                </Tooltip>
              )
            )}
          </div>

          {/* Display uploaded file, analysis buttons, and visualization for selected analysis feature */}
          {uploadedFile && (
            <div className="flex flex-col items-center w-full space-y-8">
              <Tooltip
                text="select an analysis feature"
                show={tooltipMode === "global"}
                tooltipMode={tooltipMode}
                position="bottom"
                className="w-full justify-center"
              >
                <AnalysisButtons
                  selectedInstrument={selectedInstrument}
                  selectedAnalysisFeature={selectedAnalysisFeature}
                  onAnalysisFeatureSelect={handleAnalysisFeatureSelect}
                  uploadedFile={uploadedFile}
                  audioFeatures={audioFeatures}
                  setAudioFeatures={setAudioFeatures}
                  audioUuid={audioUuid}
                  setAudioUuid={setAudioUuid}
                  uploadsEnabled={uploadsEnabled}
                />
              </Tooltip>
              {selectedAnalysisFeature && (
                <div className="flex flex-col w-full">
                  <div className="text-xl font-semibold text-lightpink mb-1">
                    {uploadedFile.name}
                  </div>
                  <div className="bg-lightgray/25 rounded-3xl w-full p-8">
                    <GraphWithWaveform
                      key={audioFeatures[selectedAnalysisFeature]?.audioUrl}
                      audioURL={
                        audioFeatures[selectedAnalysisFeature]?.audioUrl
                      }
                      featureData={
                        audioFeatures[selectedAnalysisFeature]?.data || []
                      }
                      selectedAnalysisFeature={selectedAnalysisFeature}
                      audioDuration={
                        audioFeatures[selectedAnalysisFeature]?.duration
                      }
                    />
                  </div>
                  <div className="flex flex-row justify-end gap-2 items-center mt-2">
                    <TertiaryButton onClick={handleDownloadRecording}>
                      download file
                    </TertiaryButton>
                    <TertiaryButton onClick={handleChangeFile}>
                      change file
                    </TertiaryButton>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        /* Show initial instrument select cards when you first load the app */
        <InstrumentSelectionCards
          instruments={instrumentButtons}
          handleInstrumentSelect={handleInstrumentSelect}
        />
      )}
    </div>
  );
};

export default Analyzer;
