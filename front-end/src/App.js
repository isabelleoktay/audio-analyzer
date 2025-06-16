import { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import NavBar from "./components/NavBar";
import Analyzer from "./pages/Analyzer";
import Testing from "./pages/Testing";

const App = () => {
  const [tooltipMode, setTooltipMode] = useState("inactive");

  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [selectedAnalysisFeature, setSelectedAnalysisFeature] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [inRecordMode, setInRecordMode] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioName, setAudioName] = useState("untitled.wav");
  const [audioURL, setAudioURL] = useState(null);
  const [audioUuid, setAudioUuid] = useState(null);
  const [audioFeatures, setAudioFeatures] = useState({});
  const [uploadsEnabled, setUploadsEnabled] = useState(false);

  const handleReset = () => {
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
                />
              }
            />
            <Route path="/testing" element={<Testing />} />
            <Route path="*" element={<div>404 Not Found</div>} />
          </Routes>
        </Layout>
      </Router>
    </div>
  );
};

export default App;
