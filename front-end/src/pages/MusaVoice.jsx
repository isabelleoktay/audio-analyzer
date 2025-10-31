import { useState, useEffect } from "react";
import SurveySection from "../components/survey/SurveySection.jsx";
import musaVoiceSurveyConfig from "../data/musaVoiceSurveyConfig.js";
// import UploadAudioCard from "../components/cards/UploadAudioCard";
import AnalysisButtons from "../components/buttons/AnalysisButtons.jsx";
import SecondaryButton from "../components/buttons/SecondaryButton.jsx";
import TertiaryButton from "../components/buttons/TertiaryButton.jsx";
import OverlayGraphWithWaveform from "../components/visualizations/OverlayGraphWithWaveform.jsx";
import SimilarityScoreCard from "../components/cards/SimilarityScoreCard";
import SelectedVocalTechniquesCard from "../components/cards/SelectedVocalTechniquesCard";
import MultiSelectCard from "../components/cards/MultiSelectCard.jsx";
import { mockInputFeatures, mockReferenceFeatures } from "../mock/index.js";

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

const MusaVoice = ({
  uploadedFile = "front-end/public/audio/development/input.wav",
  setUploadedFile,
  inRecordMode,
  setInRecordMode,
  audioBlob,
  setAudioBlob,
  inputAudioName,
  setInputAudioName,
  inputAudioURL = "front-end/public/audio/development/input.wav",
  setInputAudioURL,
  referenceAudioURL = "front-end/public/audio/development/reference.wav",
  setReferenceAudioURL,
  audioUuid,
  setAudioUuid,
  uploadsEnabled,
  tooltipMode,
  setUploadsEnabled,
}) => {
  const [showIntro, setShowIntro] = useState(true);
  const [showSurvey, setShowSurvey] = useState(true);
  const [showUploadAudio, setShowUploadAudio] = useState(false);
  const [analyseAudio, setAnalyseAudio] = useState(false);
  const [selectedTechniques, setSelectedTechniques] = useState([]);
  const [selectedAnalysisFeature, setSelectedAnalysisFeature] =
    useState("vocal tone");
  const [inputAudioFeatures, setInputAudioFeatures] =
    useState(mockInputFeatures);
  const [referenceAudioFeatures, setReferenceAudioFeatures] = useState(
    mockReferenceFeatures
  );
  const [selectedModel, setSelectedModel] = useState("CLAP");

  useEffect(() => {
    // Hide intro and show survey after 2 seconds
    const timer = setTimeout(() => setShowIntro(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const handleSubmitSurvey = (answers) => {
    console.log("Survey answers:", answers);
    setShowSurvey(false);
    setShowUploadAudio(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleAnalyzeNewRecording = (file) => {
    setShowUploadAudio(true);
    setAnalyseAudio(false);
  };

  const handleSubmitAudio = (audioBlob) => {
    console.log("Audio:", audioBlob);
  };

  const handleAnalysisFeatureSelect = (feature) => {
    setSelectedAnalysisFeature(feature);
  };

  const featureHasModels = ["vocal tone", "pitch mod."].includes(
    selectedAnalysisFeature
  );

  const inputData =
    (inputAudioFeatures &&
      inputAudioFeatures[selectedAnalysisFeature] &&
      (featureHasModels
        ? inputAudioFeatures[selectedAnalysisFeature][selectedModel]?.data
        : inputAudioFeatures[selectedAnalysisFeature]?.data)) ||
    null;

  const referenceData =
    (referenceAudioFeatures &&
      referenceAudioFeatures[selectedAnalysisFeature] &&
      (featureHasModels
        ? referenceAudioFeatures[selectedAnalysisFeature][selectedModel]?.data
        : referenceAudioFeatures[selectedAnalysisFeature]?.data)) ||
    null;

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
      ) : showUploadAudio ? (
        <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-8">
          <MultiSelectCard
            question="Select Target Vocal Techniques:"
            options={[
              "Vibrato",
              "Straight",
              "Trill",
              "Trillo",
              "Breathy tone",
              "Belting tone",
              "Spoken tone",
              "Inhaled singing",
              "Vocal fry",
            ]}
            allowOther={false}
            background_colour="bg-white/10"
            onChange={(selected) => setSelectedTechniques(selected)}
          />
          <div className="pt-8">
            <TertiaryButton
              onClick={() => {
                setShowUploadAudio(false);
                setAnalyseAudio(true);
              }}
              className="bg-darkpink/70 hover:bg-darkpink/100"
            >
              Proceed to Audio Analysis
            </TertiaryButton>
          </div>
        </div>
      ) : analyseAudio && uploadedFile ? (
        <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-8 pt-20">
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
            <div className="flex flex-col w-full lg:w-fit pt-6 space-y-4">
              <div className="text-xl font-semibold text-lightpink mb-1">
                {uploadedFile.name}
              </div>

              <div className="bg-lightgray/25 rounded-3xl w-full p-4 lg:p-8 overflow-x-auto lg:overflow-x-visible">
                <div className="w-full lg:min-w-[800px]">
                  {inputData && referenceData ? (
                    <OverlayGraphWithWaveform
                      inputAudioURL={inputAudioURL}
                      referenceAudioURL={referenceAudioURL}
                      inputFeatureData={inputData}
                      referenceFeatureData={referenceData}
                      selectedAnalysisFeature={selectedAnalysisFeature}
                      inputAudioDuration={
                        inputAudioFeatures[selectedAnalysisFeature]?.duration
                      }
                      selectedModel={selectedModel}
                      tooltipMode={tooltipMode}
                    />
                  ) : (
                    <div className="text-center text-lightgray/70 py-8">
                      Loading feature data...
                    </div>
                  )}
                </div>

                <div className="pt-4 pb-4 flex flex-col lg:flex-row gap-4 w-full">
                  {/* Left: the two cards side by side */}
                  <div className="flex gap-4">
                    <SimilarityScoreCard similarityScore={50.3} />
                    <div className="gap-4">
                      <SelectedVocalTechniquesCard
                        selectedTechniques={selectedTechniques}
                      />
                    </div>
                  </div>

                  {/* Right: buttons aligned to bottom */}
                  <div className="flex flex-col items-end ml-auto mt-auto gap-2">
                    {["vocal tone", "pitch mod."].includes(
                      selectedAnalysisFeature?.toLowerCase()
                    ) && (
                      <div>
                        <MultiSelectCard
                          question="Select Model:"
                          options={["CLAP", "Whisper"]}
                          allowOther={false}
                          background_colour="bg-white/10"
                          onChange={(selected) => setSelectedModel(selected)}
                          isMultiSelect={false}
                          showToggle={false}
                          miniVersion={true}
                          selected={selectedModel}
                        />
                      </div>
                    )}
                    <SecondaryButton
                      onClick={() => handleAnalyzeNewRecording()}
                    >
                      Analyse new audio
                    </SecondaryButton>
                    <SecondaryButton
                      onClick={() => handleAnalyzeNewRecording()}
                      className="from-warmyellow to-darkpink"
                    >
                      New reference audio
                    </SecondaryButton>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
};

export default MusaVoice;
