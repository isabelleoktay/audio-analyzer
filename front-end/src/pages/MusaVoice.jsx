import { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import SurveySection from "../components/survey/SurveySection.jsx";
import { uploadMusaVoiceSessionData, cleanupTempFiles } from "../utils/api.js";
import musaVoiceSurveyConfig from "../data/musaVoiceSurveyConfig.js";
import MusaUploadAudioSection from "../components/sections/MusaAudioUploadSection.jsx";
import { AnalysisButtons, SecondaryButton } from "../components/buttons";
import OverlayGraphWithWaveform from "../components/visualizations/OverlayGraphWithWaveform.jsx";
import SimilarityScoreCard from "../components/cards/SimilarityScoreCard";
import SelectedVocalTechniquesCard from "../components/cards/SelectedVocalTechniquesCard";

/**
 * The `MusaVoice` component is the main page for analyzing vocal audio files.
 * It allows users to upload or record audio, and analyze features such as pitch or dynamics after completing a survey.
 *
 * @component
 * @param {Object} props - The props passed to the component.
 * @param {boolean} props.uploadsEnabled - Whether uploads are enabled.
 * @param {string} props.tooltipMode - The mode for displaying tooltips.
 */

const MusaVoice = ({ uploadsEnabled, setUploadsEnabled, tooltipMode }) => {
  const [showIntro, setShowIntro] = useState(true);
  const [showSurvey, setShowSurvey] = useState(true);
  const [showUploadAudio, setShowUploadAudio] = useState(false);
  const [analyzeAudio, setAnalyzeAudio] = useState(false);
  const [selectedTechniques, setSelectedTechniques] = useState([]);
  const [selectedVoiceType, setSelectedVoiceType] = useState([]);
  const [answers, setAnswers] = useState({});
  const [referenceAudioData, setReferenceAudioData] = useState(null);
  const [referenceAudioSource, setReferenceAudioSource] = useState(null);
  const [userAudioData, setUserAudioData] = useState(null);
  const [userAudioSource, setUserAudioSource] = useState(null);

  const [selectedAnalysisFeature, setSelectedAnalysisFeature] = useState(null);
  const [inputAudioFeatures, setInputAudioFeatures] = useState({});
  const [referenceAudioFeatures, setReferenceAudioFeatures] = useState({});
  const [inputAudioUuid, setInputAudioUuid] = useState(() => uuidv4());
  const [similarityScore, setSimilarityScore] = useState(null);

  const [selectedModel, setSelectedModel] = useState("CLAP");
  const [sessionId, setSessionId] = useState(null);
  const [userToken, setUserToken] = useState(null);

  useEffect(() => {
    // Hide intro and show survey after 2 seconds
    const timer = setTimeout(() => setShowIntro(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // disable enabling uploads from main application
    setUploadsEnabled(false);
  }, [setUploadsEnabled]);

  useEffect(() => {
    // Always generate a new sessionId when the page/component mounts
    const newSessionId = uuidv4();
    sessionStorage.setItem("musaVoiceSessionId", newSessionId);
    setSessionId(newSessionId);

    // Get user token from localStorage
    const token = localStorage.getItem("audio_analyzer_token");
    setUserToken(token);
  }, []);

  const handleSubmitSurvey = async (answers) => {
    // console.log("Survey answers:", answers);
    setAnswers((prev) => ({ ...prev, answers }));
    setShowSurvey(false);
    setShowUploadAudio(true);
    window.scrollTo({ top: 0, behavior: "smooth" });

    // console.log("Survey answers:", answers);

    try {
      const sessionData = {
        sessionId: sessionId,
        userToken: userToken,
        surveyAnswers:
          answers && Object.keys(answers).length > 0 ? answers : {},
        timestamp: new Date().toISOString(),
        type: "musaVoice",
      };

      await uploadMusaVoiceSessionData(sessionData);
      console.log("MusaVoice session data uploaded successfully");

      setShowSurvey(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      console.error("Error uploading MusaVoice session data:", error);
    }
  };

  const handleAnalyzeNewRecording = () => {
    cleanupTempFiles(true);
    setSelectedAnalysisFeature(null);
    setShowUploadAudio(true);
    setAnalyzeAudio(false);
    setInputAudioFeatures({});
    setReferenceAudioFeatures({});
    setInputAudioUuid(uuidv4());
    setSimilarityScore(null);
    setSelectedModel("CLAP");
    setUserAudioData(null);
    setReferenceAudioData(null);
    setUserAudioSource(null);
    setReferenceAudioSource(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleAnalysisFeatureSelect = (feature) => {
    setSimilarityScore(null);
    setSelectedAnalysisFeature(feature);
  };

  const featureHasModels = ["vocal tone", "pitch mod."].includes(
    selectedAnalysisFeature
  );

  // showIntro ? (
  //       <h1 className="text-5xl font-bold text-lightpink animate-zoomIn">
  //         Welcome to MusaVoice!
  //       </h1>
  //     ) :

  const getAudioFileOrBlob = (audioData) => {
    if (!audioData) return null;

    if (audioData.source === "upload" && audioData.file) {
      return audioData.file;
    }
    if (audioData.source === "record" && audioData.blob) {
      // For recorded audio, the backend needs the Blob object
      return audioData.blob;
    }
    return null;
  };

  const userFileOrBlob = getAudioFileOrBlob(userAudioData);
  const referenceFileOrBlob = getAudioFileOrBlob(referenceAudioData);

  return (
    <div className="flex items-center justify-center min-h-screen">
      {showSurvey ? (
        <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
          <SurveySection
            config={musaVoiceSurveyConfig}
            onSubmit={handleSubmitSurvey}
          />
        </div>
      ) : showUploadAudio ? (
        <MusaUploadAudioSection
          onProceed={({
            userAudioData,
            referenceAudioData,
            selectedVoiceType,
            selectedTechniques,
            userAudioSource,
            referenceAudioSource,
          }) => {
            // Your logic to proceed to analysis
            cleanupTempFiles(true);
            setShowUploadAudio(false);
            setAnalyzeAudio(true);
            setUserAudioData(userAudioData);
            setReferenceAudioData(referenceAudioData);
            setSelectedVoiceType(selectedVoiceType);
            setSelectedTechniques(selectedTechniques);
            setUserAudioSource(userAudioSource);
            setReferenceAudioSource(referenceAudioSource);
          }}
        />
      ) : analyzeAudio && userFileOrBlob && referenceFileOrBlob ? (
        <div className="flex flex-col h-auto items-center justify-center min-h-screen text-lightgray px-8 pt-20">
          <AnalysisButtons
            selectedInstrument={"voice"}
            selectedAnalysisFeature={selectedAnalysisFeature}
            onAnalysisFeatureSelect={handleAnalysisFeatureSelect}
            inputFile={userFileOrBlob}
            referenceFile={referenceFileOrBlob}
            inputAudioFeatures={inputAudioFeatures}
            setInputAudioFeatures={setInputAudioFeatures}
            referenceAudioFeatures={referenceAudioFeatures}
            setReferenceAudioFeatures={setReferenceAudioFeatures}
            inputAudioUuid={inputAudioUuid}
            setInputAudioUuid={setInputAudioUuid}
            uploadsEnabled={uploadsEnabled}
            voiceType={selectedVoiceType}
            musaVoiceSessionId={sessionId}
          />

          {selectedAnalysisFeature && (
            <div className="flex flex-col w-full lg:w-fit pt-6 space-y-1">
              <div className="text-xl font-semibold text-lightpink">
                {userAudioData?.file?.name ||
                  userAudioData?.name ||
                  "Input Audio"}
              </div>

              <div className="bg-lightgray/25 rounded-3xl w-full p-4 lg:p-8">
                <div className="w-full lg:min-w-[800px]">
                  <OverlayGraphWithWaveform
                    inputAudioURL={
                      inputAudioFeatures[selectedAnalysisFeature]?.audioUrl
                    }
                    referenceAudioURL={
                      referenceAudioFeatures[selectedAnalysisFeature]?.audioUrl
                    }
                    inputFeatureData={
                      (featureHasModels
                        ? inputAudioFeatures[selectedAnalysisFeature]?.data?.[
                            selectedModel
                          ]
                        : inputAudioFeatures[selectedAnalysisFeature]?.data) ||
                      []
                    }
                    referenceFeatureData={
                      (featureHasModels
                        ? referenceAudioFeatures[selectedAnalysisFeature]
                            ?.data?.[selectedModel]
                        : referenceAudioFeatures[selectedAnalysisFeature]
                            ?.data) || []
                    }
                    selectedAnalysisFeature={selectedAnalysisFeature}
                    selectedVoiceType={selectedVoiceType}
                    inputAudioDuration={
                      inputAudioFeatures[selectedAnalysisFeature]?.duration
                    }
                    referenceAudioDuration={
                      referenceAudioFeatures[selectedAnalysisFeature]?.duration
                    }
                    tooltipMode={tooltipMode}
                    selectedModel={selectedModel}
                    setSelectedModel={setSelectedModel}
                    similarityScore={similarityScore}
                    setSimilarityScore={setSimilarityScore}
                  />
                </div>

                <div className="pt-4 pb-4 flex flex-col lg:flex-row gap-4 w-full">
                  {/* Left: the two cards side by side */}
                  <div className="flex gap-4">
                    <SimilarityScoreCard similarityScore={similarityScore} />
                    <div className="gap-4">
                      <SelectedVocalTechniquesCard
                        selectedTechniques={selectedTechniques}
                      />
                    </div>
                  </div>

                  {/* Right: buttons aligned to bottom */}
                  <div className="flex flex-col items-end ml-auto mt-auto gap-2">
                    <SecondaryButton onClick={handleAnalyzeNewRecording}>
                      analyze new audio
                    </SecondaryButton>
                    {/* <SecondaryButton
                      onClick={() => handleAnalyzeNewRecording()}
                      className="from-warmyellow/80 to-darkpink/80"
                    >
                      new reference audio
                    </SecondaryButton> */}
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
