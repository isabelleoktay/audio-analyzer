// import AudioAnalyzer from "./pages/AudioAnalyzer";
import { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import NavBar from "./components/NavBar";
import Analyzer from "./pages/Analyzer";

const App = () => {
  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [selectedAnalysisFeature, setSelectedAnalysisFeature] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [inRecordMode, setInRecordMode] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioName, setAudioName] = useState("untitled.wav");
  const [audioURL, setAudioURL] = useState(null);
  const [audioFeatures, setAudioFeatures] = useState({});
  const [sampleRate, setSampleRate] = useState(null);

  return (
    <div className="min-h-screen bg-radial from-bluegray to-blueblack">
      <Router>
        <Layout>
          <NavBar
            setSelectedInstrument={setSelectedInstrument}
            setUploadedFile={setUploadedFile}
            setInRecordMode={setInRecordMode}
            setAudioBlob={setAudioBlob}
            setAudioName={setAudioName}
            setAudioURL={setAudioURL}
            setSelectedAnalysisFeature={setSelectedAnalysisFeature}
            setAudioFeatures={setAudioFeatures}
            setSampleRate={setSampleRate}
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
                  sampleRate={sampleRate}
                  setSampleRate={setSampleRate}
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
