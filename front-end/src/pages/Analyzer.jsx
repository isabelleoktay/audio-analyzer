import { startTransition, useEffect } from "react";
import InstrumentSelectionCards from "../components/cards/InstrumentSelectionCards.jsx";
import InstrumentButton from "../components/buttons/InstrumentButton.jsx";
import RecordAudioSection from "../components/sections/RecordAudioSection.jsx";
import FileUploadSection from "../components/sections/FileUploadSection.jsx";
import AnalysisButtons from "../components/buttons/AnalysisButtons.jsx";
import GraphWithWaveform from "../components/visualizations/GraphWithWaveform.jsx";
import TertiaryButton from "../components/buttons/TertiaryButton.jsx";
import Tooltip from "../components/text/Tooltip.jsx";
import { instrumentButtons } from "../config/instrumentButtons.js";

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
  testingEnabled,
  setTestingEnabled,
  subjectAnalysisCount,
  setSubjectAnalysisCount,
  subjectId,
  testingPart,
  setSubjectAnalyses,
  tooltipMode,
}) => {
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
        setAudioUuid(null);
      }
    });
  };

  const handleFileUpload = (file) => {
    setUploadedFile(file);
    setAudioURL(URL.createObjectURL(file));
  };

  const handleSwitchToRecordMode = (e) => {
    e.stopPropagation();
    setInRecordMode(true);
  };

  const handleAnalysisFeatureSelect = (feature) => {
    setSelectedAnalysisFeature(feature);
  };

  const handleDownloadRecording = () => {
    const link = document.createElement("a");
    link.href = audioURL;
    link.download = audioName;
    link.click();
  };

  const handleChangeFile = () => {
    if (testingEnabled) {
      setSubjectAnalyses((prev) => ({
        ...prev,
        [testingPart]: {
          ...((prev && prev[testingPart]) || {}),
          [audioName]: {
            instrument: selectedInstrument,
            audioUuid: audioUuid,
            audioFeatures: audioFeatures,
          },
        },
      }));
    }
    setSelectedAnalysisFeature(null);
    setUploadedFile(null);
    setAudioBlob(null);
    const newsubjectAnalysisCount = subjectAnalysisCount + 1;
    setSubjectAnalysisCount(newsubjectAnalysisCount);

    let newAudioName;
    if (testingEnabled) {
      newAudioName = `subject-${subjectId}-${testingPart}-${newsubjectAnalysisCount}.wav`;
    } else {
      newAudioName = "untitled.wav";
    }
    setAudioName(newAudioName);

    setAudioURL(null);
    setAudioFeatures({});
    if (testingEnabled) {
      setInRecordMode(true);
    } else {
      setInRecordMode(false);
    }
  };

  useEffect(() => {
    console.log("Audio features updated:");
    console.log(audioFeatures);
  }, [audioFeatures]);

  // useEffect(() => {
  //   const newAudioName = `subject-${subjectId}-${testingPart}-${subjectAnalysisCount}.wav`;
  //   setAudioName(newAudioName);
  // }, [subjectAnalysisCount, setAudioName, subjectId, testingPart]);

  return (
    <div className="flex flex-col items-center min-h-[calc(100vh-4rem)]">
      {selectedInstrument ? (
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
              testingEnabled={testingEnabled}
              setTestingEnabled={setTestingEnabled}
            />
          ) : (
            <div className="flex flex-col items-center w-full">
              {!selectedAnalysisFeature && (
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
                    testingEnabled={testingEnabled}
                  />
                </Tooltip>
              )}
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
                        />
                      </div>
                      <div className="flex flex-row justify-end gap-2 items-center mt-2">
                        <TertiaryButton onClick={handleDownloadRecording}>
                          download file
                        </TertiaryButton>
                        <TertiaryButton onClick={handleChangeFile}>
                          {testingEnabled
                            ? "submit and analyze next"
                            : "change file"}
                        </TertiaryButton>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        <InstrumentSelectionCards
          instruments={instrumentButtons}
          handleInstrumentSelect={handleInstrumentSelect}
        />
      )}
    </div>
  );
};

export default Analyzer;
