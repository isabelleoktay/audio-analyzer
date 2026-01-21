import { startTransition, useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";

import InstrumentSelectionCards from "../components/cards/InstrumentSelectionCards.jsx";
import InstrumentButton from "../components/buttons/InstrumentButton.jsx";
import RecordAudioSection from "../components/sections/RecordAudioSection.jsx";
import FileUploadSection from "../components/sections/FileUploadSection.jsx";
import AnalysisButtons from "../components/buttons/AnalysisButtons.jsx";
import OverlayGraphWithWaveform from "../components/visualizations/OverlayGraphWithWaveform.jsx";
import TertiaryButton from "../components/buttons/TertiaryButton.jsx";
import Tooltip from "../components/text/Tooltip.jsx";
import ConsentModal from "../components/modals/ConsentModal.jsx";
import { instrumentButtons } from "../config/instrumentButtons.js";

/**
 * The `Analyzer` component is the main page for analyzing audio files.
 * It allows users to select an instrument, upload or record audio, and analyze features such as pitch or dynamics.
 *
 * @component
 * @param {Object} props - The props passed to the component.
 * @param {string} props.selectedInstrument - The currently selected instrument.
 * @param {Function} props.setSelectedInstrument - Function to update the selected instrument.
 * @param {File} props.inputFile - The uploaded audio file.
 * @param {Function} props.setInputFile - Function to update the uploaded file.
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
 * @param {Object} props.inputAudioFeatures - Extracted features from the audio.
 * @param {Function} props.setInputAudioFeatures - Function to update the audio features.
 * @param {string} props.inputAudioUuid - A unique identifier for the audio session.
 * @param {Function} props.setInputAudioUuid - Function to update the audio UUID.
 * @param {boolean} props.uploadsEnabled - Whether uploads are enabled.
 * @param {string} props.tooltipMode - The mode for displaying tooltips.
 * @returns {JSX.Element} The rendered `Analyzer` component.
 */
const Analyzer = ({
  selectedInstrument,
  setSelectedInstrument,
  inputFile,
  setInputFile,
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
  inputAudioFeatures,
  setInputAudioFeatures,
  inputAudioUuid,
  setInputAudioUuid,
  uploadsEnabled,
  tooltipMode,
  setUploadsEnabled,
}) => {
  // Initialize consent state based on localStorage to prevent flashing
  const [hasConsented, setHasConsented] = useState(() => {
    return localStorage.getItem("audioAnalyzerConsent") === "true";
  });

  const [showConsentModal, setShowConsentModal] = useState(() => {
    return localStorage.getItem("audioAnalyzerConsent") !== "true";
  });

  const [selectedModel, setSelectedModel] = useState("CLAP");
  const [audioData, setAudioData] = useState(null);
  const [audioSource, setAudioSource] = useState(null);

  const featureHasModels = ["vocal tone", "pitch mod."].includes(
    selectedAnalysisFeature,
  );

  const getAudioFileOrBlob = (data, source) => {
    if (!data || !source) return null;
    return data;
  };

  const inputFileOrBlob = getAudioFileOrBlob(audioData, audioSource);

  // Set selectedModel based on available models in the response
  useEffect(() => {
    if (!featureHasModels) return;

    const featureData = inputAudioFeatures[selectedAnalysisFeature]?.data;
    if (!featureData || typeof featureData !== "object") return;

    const availableModels = Object.keys(featureData).filter(
      (key) =>
        ["CLAP", "Whisper"].includes(key) && featureData[key]?.length > 0,
    );

    if (availableModels.length > 0) {
      // Set to first available model if current selection isn't available
      if (!availableModels.includes(selectedModel)) {
        setSelectedModel(availableModels[0]);
      }
    }
  }, [
    selectedAnalysisFeature,
    inputAudioFeatures,
    featureHasModels,
    selectedModel,
  ]);

  // Handle consent response
  const handleConsent = (agreed) => {
    if (agreed) {
      setHasConsented(true);
      setShowConsentModal(false);
      localStorage.setItem("audioAnalyzerConsent", "true");
    } else {
      setHasConsented(false);
      setShowConsentModal(true);
    }
  };

  // Handles the selection of an instrument.
  // Resets audio-related state if audio features are already present.
  const handleInstrumentSelect = (instrument) => {
    startTransition(() => {
      setSelectedInstrument(instrument);
      if (Object.keys(inputAudioFeatures).length > 0) {
        console.log("Resetting audio features");
        setInputFile(null);
        setAudioBlob(null);
        setAudioData(null);
        setAudioSource(null);
        setAudioName("untitled.wav");
        setAudioURL(null);
        setInputAudioFeatures({});
        setSelectedAnalysisFeature(null);
        setInputAudioUuid(() => uuidv4());
      }
    });
  };

  // Handles the upload of an audio file.
  // Updates the uploaded file and generates a URL for it.
  const handleFileUpload = (file) => {
    setInputFile(file);
    setAudioURL(URL.createObjectURL(file));
    setAudioData(file);
    setAudioSource("upload");
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
    setInputFile(null);
    setAudioBlob(null);
    setAudioData(null);
    setAudioSource(null);

    const newAudioName = "untitled.wav";
    setAudioName(newAudioName);

    setAudioURL(null);
    setInputAudioFeatures({});
    setInRecordMode(false);
    setInputAudioUuid(() => uuidv4());
  };

  useEffect(() => {
    // enable enabling uploads in main application
    setUploadsEnabled(true);
  }, [setUploadsEnabled]);

  return (
    <div className="min-h-screen w-full">
      <ConsentModal isOpen={showConsentModal} onConsent={handleConsent} />

      <div
        className={`flex flex-col w-full items-center min-h-[calc(100vh-4rem)] ${
          hasConsented ? "" : "pointer-events-none opacity-50"
        }`}
      >
        {selectedInstrument ? (
          /* Show smaller intrument selection buttons at the top of screen if an instrument is already selected. */
          <div className="mt-20 lg:mt-28 mb-8 w-full lg:w-1/2">
            <Tooltip
              text="select an instrument to analyze"
              show={tooltipMode === "global"}
              tooltipMode={tooltipMode}
              className="w-full"
            >
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 lg:gap-6 mb-8 lg:mb-16">
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
                  setInputFile={setInputFile}
                  setInRecordMode={setInRecordMode}
                  audioBlob={audioBlob}
                  setAudioBlob={setAudioBlob}
                  audioName={audioName}
                  setAudioName={setAudioName}
                  audioURL={audioURL}
                  setAudioURL={setAudioURL}
                  handleDownloadRecording={handleDownloadRecording}
                  setAudioData={setAudioData}
                  setAudioSource={setAudioSource}
                  className="w-full "
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
                      inputFile={inputFile}
                      className="mb-8 w-full"
                    />
                  </Tooltip>
                )
              )}
            </div>

            {/* Display uploaded file, analysis buttons, and visualization for selected analysis feature */}
            {inputFileOrBlob && (
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
                    inputFileOrBlob={inputFileOrBlob}
                    inputAudioFeatures={inputAudioFeatures}
                    setInputAudioFeatures={setInputAudioFeatures}
                    inputAudioUuid={inputAudioUuid}
                    setInputAudioUuid={setInputAudioUuid}
                    uploadsEnabled={uploadsEnabled}
                    // monitorResources={false}
                  />
                </Tooltip>
                {selectedAnalysisFeature && (
                  <div className="flex flex-col w-full lg:w-fit">
                    <div className="text-xl font-semibold text-lightpink mb-1">
                      {inputFile?.name || audioName}
                    </div>
                    <div className="bg-lightgray/25 rounded-3xl w-full p-4 lg:p-8 overflow-x-auto lg:overflow-x-visible">
                      {/* Add overflow-x-auto on mobile only */}
                      <div className="w-full lg:min-w-[800px]">
                        {/* Ensure 800px minimum width */}
                        <OverlayGraphWithWaveform
                          key={
                            inputAudioFeatures[selectedAnalysisFeature]
                              ?.audioUrl
                          }
                          inputAudioURL={
                            inputAudioFeatures[selectedAnalysisFeature]
                              ?.audioUrl
                          }
                          inputFeatureData={
                            inputAudioFeatures[selectedAnalysisFeature]?.data ||
                            []
                          }
                          selectedAnalysisFeature={selectedAnalysisFeature}
                          inputAudioDuration={
                            inputAudioFeatures[selectedAnalysisFeature]
                              ?.duration
                          }
                          tooltipMode={tooltipMode}
                          selectedModel={selectedModel}
                          setSelectedModel={setSelectedModel}
                        />
                      </div>
                    </div>
                    <div className="flex w-full flex-col lg:flex-row justify-end gap-2 items-center mt-2">
                      <TertiaryButton
                        onClick={handleDownloadRecording}
                        className="w-full lg:w-auto"
                      >
                        download file
                      </TertiaryButton>
                      <TertiaryButton
                        onClick={handleChangeFile}
                        className="w-full lg:w-auto"
                      >
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
    </div>
  );
};

export default Analyzer;
