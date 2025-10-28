import { useState, useEffect } from "react";
import SurveySection from "../components/survey/SurveySection.jsx";
import musaVoiceSurveyConfig from "../data/musaVoiceSurveyConfig.js";
// import UploadAudioCard from "../components/cards/UploadAudioCard";
import AnalysisButtons from "../components/buttons/AnalysisButtons.jsx";
import TertiaryButton from "../components/buttons/TertiaryButton.jsx";
import OverlayGraphWithWaveform from "../components/visualizations/OverlayGraphWithWaveform.jsx";
import {
  mockInputFeatures,
  mockReferenceFeatures,
} from "../mock/mockFeatureData";

/**
 * The `MusaVoice` component is the main page for analyzing vocal audio files.
 * It allows users to upload or record audio, and analyze features such as pitch or dynamics after completing a survey.
 *
 * @component
 * @param {Object} props - The props passed to the component.
 * @param {File} props.uploadedFile - The uploaded audio file.
 * @param {Function} props.setUploadedFile - Function to update the uploaded file.
 * @param {boolean} props.inRecordMode - Whether a user is recording audio to be uploaded.
 * @param {Function} props.setInRecordMode - Function to toggle recording mode.
 * @param {Blob} props.audioBlob - The audio blob data for recording.
 * @param {Function} props.setAudioBlob - Function to update the audio blob.
 * @param {string} props.audioName - The name of the audio file.
 * @param {Function} props.setAudioName - Function to update the audio file name.
 * @param {string} props.inputAudioURL - The URL of the uploaded or recorded input audio.
 * @param {Function} props.setInputAudioURL - Function to update the input audio URL.
 * @param {string} props.referenceAudioURL - The URL of the uploaded or recorded reference audio.
 * @param {Function} props.setReferenceAudioURL - Function to update the reference audio URL.
 * @param {string} props.selectedAnalysisFeature - The selected audio analysis feature.
 * @param {Function} props.setSelectedAnalysisFeature - Function to update the selected analysis feature.
 * @param {Object} props.inputAudioFeatures - Extracted features from the input audio.
 * @param {Function} props.setInputAudioFeatures - Function to update the input audio features.
 * @param {Object} props.referenceAudioFeatures - Extracted features from the input audio.
 * @param {Function} props.setReferenceAudioFeatures - Function to update the input audio features.
 * @param {string} props.audioUuid - A unique identifier for the audio session.
 * @param {Function} props.setAudioUuid - Function to update the audio UUID.
 * @param {boolean} props.uploadsEnabled - Whether uploads are enabled.
 * @param {string} props.tooltipMode - The mode for displaying tooltips.
 * @returns {JSX.Element} The rendered `Analyzer` component.
 */

const MusaVoice = (
  uploadedFile,
  setUploadedFile,
  inRecordMode,
  setInRecordMode,
  audioBlob,
  setAudioBlob,
  audioName,
  setAudioName,
  inputAudioURL,
  setInputAudioURL,
  referenceAudioURL,
  setReferenceAudioURL,
  selectedAnalysisFeature = "vocal tone",
  setSelectedAnalysisFeature,
  inputAudioFeatures = mockInputFeatures,
  setInputAudioFeatures,
  referenceAudioFeatures = mockReferenceFeatures,
  setReferenceAudioFeatures,
  audioUuid,
  setAudioUuid,
  uploadsEnabled,
  tooltipMode,
  setUploadsEnabled
) => {
  const [showIntro, setShowIntro] = useState(true);
  const [showSurvey, setShowSurvey] = useState(true);

  useEffect(() => {
    // Hide intro and show survey after 2 seconds
    const timer = setTimeout(() => setShowIntro(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const handleSubmitSurvey = (answers) => {
    console.log("Survey answers:", answers);
    setShowSurvey(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleSubmitAudio = (audioBlob) => {
    console.log("Audio:", audioBlob);
  };

  const handleAnalysisFeatureSelect = (feature) => {
    setSelectedAnalysisFeature(feature);
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
      {showIntro ? (
        <h1 className="text-5xl font-bold text-lightpink animate-zoomIn">
          Welcome to MusaVoice!
        </h1>
      ) : showSurvey ? (
        <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
          <SurveySection
            config={musaVoiceSurveyConfig}
            onSubmit={handleSubmitSurvey}
          />
        </div>
      ) : (
        <div className="flex flex-col items-center w-full space-y-8">
          {/* <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-8">
            <UploadAudioCard 
              label="Reference Audio" 
              onSubmit={handleSubmitAudio}
            />
          </div> */}
          <AnalysisButtons
            selectedInstrument={"voice"}
            selectedAnalysisFeature={selectedAnalysisFeature}
            onAnalysisFeatureSelect={handleAnalysisFeatureSelect}
            uploadedFile={uploadedFile}
            audioFeatures={inputAudioFeatures}
            setAudioFeatures={setInputAudioFeatures}
            audioUuid={audioUuid}
            setAudioUuid={setAudioUuid}
            uploadsEnabled={uploadsEnabled}
          />
          {selectedAnalysisFeature && (
            <div className="flex flex-col w-full lg:w-fit">
              <div className="text-xl font-semibold text-lightpink mb-1">
                {uploadedFile.name}
              </div>
              <div className="bg-lightgray/25 rounded-3xl w-full p-4 lg:p-8 overflow-x-auto lg:overflow-x-visible">
                {/* Add overflow-x-auto on mobile only */}
                <div className="w-full lg:min-w-[800px]">
                  {/* Ensure 800px minimum width */}
                  <OverlayGraphWithWaveform
                    inputAudioURL={
                      inputAudioFeatures[selectedAnalysisFeature]?.inputAudioUrl
                    }
                    referenceAudioURL={
                      referenceAudioFeatures[selectedAnalysisFeature]
                        ?.referenceAudioUrl
                    }
                    inputFeatureData={
                      inputAudioFeatures[selectedAnalysisFeature]?.data || []
                    }
                    referenceFeatureData={
                      referenceAudioFeatures[selectedAnalysisFeature]?.data ||
                      []
                    }
                    selectedAnalysisFeature={selectedAnalysisFeature}
                    inputAudioDuration={
                      inputAudioFeatures[selectedAnalysisFeature]?.duration
                    }
                    tooltipMode={tooltipMode}
                  />
                </div>
              </div>
              <div className="flex w-full flex-col lg:flex-row justify-end gap-2 items-center mt-2">
                {/* <TertiaryButton
                  onClick={handleDownloadRecording}
                  className="w-full lg:w-auto"
                >
                  download file
                </TertiaryButton> */}
                {/* <TertiaryButton
                  onClick={handleChangeFile}
                  className="w-full lg:w-auto"
                >
                  change file
                </TertiaryButton> */}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MusaVoice;
