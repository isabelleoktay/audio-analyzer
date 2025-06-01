import { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import NavBar from "./components/NavBar";
import Analyzer from "./pages/Analyzer";
import Testing from "./pages/Testing";
import useLocalStorageState from "./hooks/useLocalStorageState";

const App = () => {
  const [tooltipMode, setTooltipMode] = useState("inactive");

  const [testingEnabled, setTestingEnabled] = useLocalStorageState(
    "testingEnabled",
    false,
    true
  );
  const [selectedInstrument, setSelectedInstrument] = useLocalStorageState(
    "selectedInstrument",
    null,
    testingEnabled
  );
  const [selectedAnalysisFeature, setSelectedAnalysisFeature] =
    useLocalStorageState("selectedAnalysisFeature", null, testingEnabled);
  const [uploadedFile, setUploadedFile] = useLocalStorageState(
    "uploadedFile",
    null,
    testingEnabled
  );
  const [inRecordMode, setInRecordMode] = useLocalStorageState(
    "inRecordMode",
    false,
    testingEnabled
  );
  const [audioBlob, setAudioBlob] = useLocalStorageState(
    "audioBlob",
    null,
    testingEnabled
  );
  const [audioName, setAudioName] = useLocalStorageState(
    "audioName",
    "untitled.wav",
    testingEnabled
  );
  const [audioURL, setAudioURL] = useLocalStorageState(
    "audioURL",
    null,
    testingEnabled
  );
  const [audioUuid, setAudioUuid] = useLocalStorageState(
    "audioUuid",
    null,
    testingEnabled
  );
  const [audioFeatures, setAudioFeatures] = useLocalStorageState(
    "audioFeatures",
    {},
    testingEnabled
  );
  const [uploadsEnabled, setUploadsEnabled] = useLocalStorageState(
    "uploadsEnabled",
    false,
    testingEnabled
  );
  const [subjectId, setSubjectId] = useLocalStorageState("subjectId", null);
  const [testingPart, setTestingPart] = useLocalStorageState(
    "testingPart",
    "partA",
    testingEnabled
  );
  const [subjectAnalysisCount, setSubjectAnalysisCount] = useLocalStorageState(
    "subjectAnalysisCount",
    1,
    testingEnabled
  );
  const [subjectAnalyses, setSubjectAnalyses] = useLocalStorageState(
    "subjectAnalyses",
    {},
    testingEnabled
  );

  const handleReset = () => {
    localStorage.clear();
    setSelectedInstrument(null);
    setUploadedFile(null);
    setInRecordMode(false);
    setAudioBlob(null);
    setAudioName("untitled.wav");
    setAudioURL(null);
    setSelectedAnalysisFeature(null);
    setAudioFeatures({});
    setTooltipMode("inactive");
    setAudioUuid(null);
    setUploadsEnabled(false);
    setTestingEnabled(false);
    setSubjectId(null);
    setTestingPart("partA");
    setSubjectAnalysisCount(1);
    setSubjectAnalyses({});
  };

  const resetAudioData = (resetAudioName = false) => {
    setUploadedFile(null);
    setAudioBlob(null);
    if (resetAudioName) {
      setAudioName("untitled.wav");
    }
    setSelectedAnalysisFeature(null);
    setAudioURL(null);
    setAudioFeatures({});
    setAudioUuid(null);
  };

  return (
    <div className="min-h-screen bg-radial from-bluegray to-blueblack">
      <Router>
        <Layout>
          <NavBar
            handleReset={handleReset}
            uploadsEnabled={uploadsEnabled}
            setUploadsEnabled={setUploadsEnabled}
            setTooltipMode={setTooltipMode}
            tooltipMode={tooltipMode}
          />
          <Routes>
            <Route
              path="/"
              element={
                <Analyzer
                  selectedInstrument={selectedInstrument}
                  setSelectedInstrument={setSelectedInstrument}
                  uploadedFile={uploadedFile}
                  setUploadedFile={setUploadedFile}
                  inRecordMode={inRecordMode}
                  setInRecordMode={setInRecordMode}
                  setAudioBlob={setAudioBlob}
                  audioBlob={audioBlob}
                  setAudioName={setAudioName}
                  audioName={audioName}
                  setAudioURL={setAudioURL}
                  audioURL={audioURL}
                  selectedAnalysisFeature={selectedAnalysisFeature}
                  setSelectedAnalysisFeature={setSelectedAnalysisFeature}
                  audioFeatures={audioFeatures}
                  setAudioFeatures={setAudioFeatures}
                  handleReset={handleReset}
                  tooltipMode={tooltipMode}
                  audioUuid={audioUuid}
                  setAudioUuid={setAudioUuid}
                  uploadsEnabled={uploadsEnabled}
                  testingEnabled={testingEnabled}
                  setTestingEnabled={setTestingEnabled}
                  subjectAnalysisCount={subjectAnalysisCount}
                  setSubjectAnalysisCount={setSubjectAnalysisCount}
                  subjectAnalyses={subjectAnalyses}
                  setSubjectAnalyses={setSubjectAnalyses}
                  subjectId={subjectId}
                  testingPart={testingPart}
                />
              }
            />
            <Route
              path="/testing"
              element={
                <Testing
                  testingEnabled={testingEnabled}
                  setTestingEnabled={setTestingEnabled}
                  setSubjectId={setSubjectId}
                  subjectId={subjectId}
                  testingPart={testingPart}
                  setTestingPart={setTestingPart}
                  audioName={audioName}
                  setAudioName={setAudioName}
                  setSubjectAnalysisCount={setSubjectAnalysisCount}
                  subjectAnalysisCount={subjectAnalysisCount}
                  setInRecordMode={setInRecordMode}
                  subjectAnalyses={subjectAnalyses}
                  resetAudioData={resetAudioData}
                />
              }
            />
            <Route path="*" element={<div>404 Not Found</div>} />
          </Routes>
        </Layout>
      </Router>
    </div>
  );
};

export default App;
