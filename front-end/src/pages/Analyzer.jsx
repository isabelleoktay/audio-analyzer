import { useMemo, startTransition } from "react";
import InstrumentSelectionCards from "../components/cards/InstrumentSelectionCards.jsx";
import InstrumentButton from "../components/buttons/InstrumentButton.jsx";
import RecordAudioSection from "../components/sections/RecordAudioSection.jsx";
import FileUploadSection from "../components/sections/FileUploadSection.jsx";
import AnalysisButtons from "../components/buttons/AnalysisButtons.jsx";
import GraphWithWaveform from "../components/graphs/GraphWithWaveform.jsx";
import TertiaryButton from "../components/buttons/TertiaryButton.jsx";
import { instrumentButtons } from "../config/instrumentButtons.js";

// const dynamicsData = Array.from({ length: 100 }, (_, i) => {
//   const base = 0.5 + 0.3 * Math.sin(i / 10); // Sine wave to simulate musical phrasing
//   const noise = (Math.random() - 0.5) * 0.1; // Small random variation
//   return Math.min(1, Math.max(0, base + noise)); // Clamp between 0 and 1
// });

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
  sampleRate,
  setSampleRate,
}) => {
  const highlightedSections = useMemo(() => [], []);

  const handleInstrumentSelect = (instrument) => {
    startTransition(() => {
      setSelectedInstrument(instrument);
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

  return (
    <div className="flex flex-col items-center min-h-[calc(100vh-4rem)]">
      {selectedInstrument ? (
        <div className="mt-28">
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
            />
          ) : (
            <div className="flex flex-col items-center w-full">
              {!selectedAnalysisFeature && (
                <FileUploadSection
                  handleSwitchToRecordMode={handleSwitchToRecordMode}
                  handleFileUpload={handleFileUpload}
                  uploadedFile={uploadedFile}
                  className="mb-8 w-full"
                />
              )}
              {uploadedFile && (
                <div className="flex flex-col items-center w-full space-y-8">
                  <AnalysisButtons
                    selectedInstrument={selectedInstrument}
                    selectedAnalysisFeature={selectedAnalysisFeature}
                    onAnalysisFeatureSelect={handleAnalysisFeatureSelect}
                    audioFile={uploadedFile}
                    audioFeatures={audioFeatures}
                    setAudioFeatures={setAudioFeatures}
                    sampleRate={sampleRate}
                    setSampleRate={setSampleRate}
                  />
                  {selectedAnalysisFeature && (
                    <div className="flex flex-col w-full">
                      <div className="text-xl font-semibold text-lightpink mb-1">
                        {uploadedFile.name}
                      </div>
                      <div className="bg-lightgray/25 rounded-3xl w-full p-8">
                        <GraphWithWaveform
                          key={audioURL}
                          audioURL={audioURL}
                          featureData={
                            audioFeatures[selectedAnalysisFeature] || []
                          }
                          sampleRate={44100}
                          highlightedSections={highlightedSections}
                          selectedAnalysisFeature={selectedAnalysisFeature}
                        />
                      </div>
                      <TertiaryButton
                        onClick={() => {
                          setSelectedAnalysisFeature(null);
                          setUploadedFile(null);
                          setAudioBlob(null);
                          setAudioName("untitled.wav");
                          setAudioURL(null);
                          setAudioFeatures({});
                        }}
                        className="mt-2 w-1/6 self-end"
                      >
                        change file
                      </TertiaryButton>
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
