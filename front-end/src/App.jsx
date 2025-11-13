import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";
import Layout from "./components/Layout.jsx";
import NavBar from "./components/NavBar.jsx";
import {
  Analyzer,
  Testing,
  HowToUse,
  NotFound,
  MusaVoice,
  MultiAudio,
} from "./pages";

import { cleanupTempFiles } from "./utils/api.js";
import { tokenManager } from "./utils/tokenManager.js";

/**
 * The main application component for the Audio Analyzer frontend.
 * It manages global state, routing, and layout for the application.
 *
 * @component
 * @returns {JSX.Element} The rendered App component.
 */
const App = () => {
  const [tooltipMode, setTooltipMode] = useState("inactive");

  const [selectedInstrument, setSelectedInstrument] = useState(null);
  const [selectedAnalysisFeature, setSelectedAnalysisFeature] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [inRecordMode, setInRecordMode] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioName, setAudioName] = useState("untitled.wav");
  const [audioURL, setAudioURL] = useState(null);
  const [audioUuid, setAudioUuid] = useState(() => uuidv4());
  const [audioFeatures, setAudioFeatures] = useState({});
  const [uploadsEnabled, setUploadsEnabled] = useState(true);

  // Reset the application state to its initial values.
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
    setAudioUuid(() => uuidv4());
    setUploadsEnabled(false);
  };

  // Clean up temporary files when the component unmounts.
  useEffect(() => {
    return () => {
      cleanupTempFiles();
    };
  }, []);

  useEffect(() => {
    // Initialize token when app starts
    const initializeToken = async () => {
      try {
        await tokenManager.ensureValidToken();
        console.log("App initialized with valid token");
      } catch (error) {
        console.error("Error initializing token:", error);
      }
    };

    initializeToken();
  }, []);

  return (
    <div className="relative">
      {/* Fixed background layer */}
      <div className="fixed inset-0 z-0">
        <div className="w-full min-h-screen bg-radial from-bluegray to-blueblack bg-fixed" />
      </div>

      {/* Scrollable content layer */}
      <div className="relative z-10">
        <div className="flex flex-col min-h-screen">
          <Router>
            <Layout>
              {/* Navigation bar with reset functionality and tooltip controls */}
              <NavBar
                handleReset={handleReset}
                uploadsEnabled={uploadsEnabled}
                setUploadsEnabled={setUploadsEnabled}
                setTooltipMode={setTooltipMode}
                tooltipMode={tooltipMode}
              />
              <Routes>
                {/* Main Analyzer page for audio analysis */}
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
                      setUploadsEnabled={setUploadsEnabled}
                    />
                  }
                />
                {/* Testing page for additional functionality */}
                <Route
                  path="/testing"
                  element={<Testing setUploadsEnabled={setUploadsEnabled} />}
                />
                <Route
                  path="/multi-audio"
                  element={<MultiAudio setUploadsEnabled={setUploadsEnabled} />}
                />
                <Route path="/how-to-use" element={<HowToUse />} />
                <Route path="/musa-voice" element={<MusaVoice />} />
                <Route path="/musa-feedback" element={<FeedbackForm />} />

                {/* Fallback route for undefined paths */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Layout>
          </Router>
        </div>
      </div>
    </div>
  );
};

export default App;
