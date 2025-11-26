import { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import SurveySection from "../components/survey/SurveySection.jsx";
import { uploadMusaVoiceSessionData } from "../utils/api.js";
import musaVoiceSurveyConfig from "../data/musaVoiceSurveyConfig.js";
// import UploadAudioCard from "../components/cards/UploadAudioCard";
import {
  AnalysisButtons,
  SecondaryButton,
} from "../components/buttons";
import OverlayGraphWithWaveform from "../components/visualizations/OverlayGraphWithWaveform.jsx";
import UploadAudioCard from "../components/cards/UploadAudioCard.jsx";
import SimilarityScoreCard from "../components/cards/SimilarityScoreCard";
import SelectedVocalTechniquesCard from "../components/cards/SelectedVocalTechniquesCard";
import MultiSelectCard from "../components/cards/MultiSelectCard.jsx";
import InstructionCard from "../components/cards/InstructionCard.jsx";
// import { mockInputFeatures, mockReferenceFeatures } from "../mock/index.js";

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
//   const [showIntro, setShowIntro] = useState(true);
  const [showSurvey, setShowSurvey] = useState(true);
  const [showUploadAudio, setShowUploadAudio] = useState(false);
  const [analyzeAudio, setAnalyzeAudio] = useState(false);
  const [selectedTechniques, setSelectedTechniques] = useState([]);
  const [selectedVoiceType, setSelectedVoiceType] = useState([]);
  const [answers, setAnswers] = useState({});
  const [referenceAudioSource, setReferenceAudioSource] = useState(null);
  const [referenceAudioData, setReferenceAudioData] = useState(null);
  const [userAudioSource, setUserAudioSource] = useState(null);
  const [userAudioData, setUserAudioData] = useState(null);

  const [selectedAnalysisFeature, setSelectedAnalysisFeature] = useState(null);
  const [inputAudioFeatures, setInputAudioFeatures] = useState({});
  const [referenceAudioFeatures, setReferenceAudioFeatures] = useState({});
  const [inputAudioUuid, setInputAudioUuid] = useState(() => uuidv4());
  const [similarityScore, setSimilarityScore] = useState(null);

  const [selectedModel, setSelectedModel] = useState("CLAP");
  const [sessionId, setSessionId] = useState(null);
  const [userToken, setUserToken] = useState(null);

//   useEffect(() => {
//     // Hide intro and show survey after 2 seconds
//     const timer = setTimeout(() => setShowIntro(false), 1500);
//     return () => clearTimeout(timer);
//   }, []);

  useEffect(() => {
    // disable enabling uploads from main application
    setUploadsEnabled(false);
  }, [setUploadsEnabled]);

  useEffect(() => {
    // Get or create session ID
    let currentSessionId = sessionStorage.getItem("musaVoiceSessionId");
    if (!currentSessionId) {
      currentSessionId = uuidv4();
      sessionStorage.setItem("musaVoiceSessionId", currentSessionId);
    }
    setSessionId(currentSessionId);

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
        surveyAnswers: answers,
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

  const handleAnalyzeNewRecording = (file) => {
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

  const handleAudioSourceChange = (source, type) => {
    // console.log(`${type} audio source changed to:`, source);
    if (type === "reference") {
      setReferenceAudioSource(source);
    } else if (type === "user") {
      setUserAudioSource(source);
    }
  };

  const handleAudioDataChange = (audioData, type) => {
    // console.log(`${type} audio data updated:`, audioData);
    if (type === "reference") {
      setReferenceAudioData(audioData);
    } else if (type === "user") {
      setUserAudioData(audioData);
    }
  };

  // Check if audio is ready for analysis
  const isAudioReady = (audioSource, audioData) => {
    if (!audioSource) return false;

    if (audioSource === "upload") {
      return audioData?.file !== null && audioData?.file !== undefined;
    } else if (audioSource === "record") {
      return audioData?.blob !== null && audioData?.blob !== undefined;
    }

    return false;
  };

  const isReferenceReady = isAudioReady(
    referenceAudioSource,
    referenceAudioData
  );
  const isUserReady = isAudioReady(userAudioSource, userAudioData);
  const isVoiceTypeSelected = selectedVoiceType && selectedVoiceType.length > 0;
  const isTechniquesSelected =
    selectedTechniques && selectedTechniques.length > 0;
  const isFormValid =
    isReferenceReady &&
    isUserReady &&
    isVoiceTypeSelected &&
    isTechniquesSelected;

  const handleAnalyzeClick = () => {
    if (isFormValid) {
      setShowUploadAudio(false);
      setAnalyzeAudio(true);
      //   console.log("Analyzing...");
      // Proceed with analysis
      //   console.log("Reference:", referenceAudioData);
      //   console.log("User:", userAudioData);
    }
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
    if (audioData.source === "recording" && audioData.blob) {
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
        <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-8 w-full">
          {/* <div className="flex flex-row w-full gap-8 justify-center items-center"> */}
          {/* <div className="flex flex-col gap-2 items-center mb-4 h-full w-full">
            <div className="font-bold text-5xl text-electricblue text-center">
              upload audio
            </div>
            <div className="text-lightgray text-xl w-3/4 text-center">
              the system will analyse how closely your test audio matches the
              vocal techniques in the uploaded/recorded reference audio.
            </div>
          </div> */}
          <div className="flex flex-row w-full gap-8 justify-center items-stretch min-h-[400px]">
            <div className="flex flex-col w-full gap-3">
              <InstructionCard
                stepNumber={1}
                title="upload your input and reference audios"
                description="the system will analyse how closely your input audio matches the vocal techniques in the uploaded/recorded reference audio."
              />
              <UploadAudioCard
                label="input audio"
                onAudioSourceChange={(source) =>
                  handleAudioSourceChange(source, "user")
                }
                onAudioDataChange={(data) =>
                  handleAudioDataChange(data, "user")
                }
              />
              <UploadAudioCard
                label="reference audio"
                onAudioSourceChange={(source) =>
                  handleAudioSourceChange(source, "reference")
                }
                onAudioDataChange={(data) =>
                  handleAudioDataChange(data, "reference")
                }
              />
            </div>
            <div className="self-stretch w-px bg-darkgray/70"></div>
            <div className="flex flex-col gap-3">
              <InstructionCard
                stepNumber={2}
                title="select vocal specifics"
                description="the system will use this information to calibrate the analysis."
              />
              <div className="flex flex-row w-full gap-2">
                <MultiSelectCard
                  question="select voice type of input audio:"
                  options={["bass", "tenor", "alto", "soprano"]}
                  allowOther={false}
                  background_color="bg-white/10"
                  onChange={(selected) => setSelectedVoiceType(selected)}
                />
                <MultiSelectCard
                  question="select target vocal techniques:"
                  options={[
                    "vibrato",
                    "straight",
                    "trill",
                    "trillo",
                    "breathy tone",
                    "belting tone",
                    "spoken tone",
                    "inhaled singing",
                    "vocal fry",
                  ]}
                  allowOther={false}
                  background_color="bg-white/10"
                  onChange={(selected) => setSelectedTechniques(selected)}
                />
              </div>
            </div>
          </div>
          <div className="pt-8">
            <SecondaryButton
              className={`h-fit text-xl transition-all duration-200 ${
                !isFormValid && "opacity-50 cursor-not-allowed"
              }`}
              onClick={handleAnalyzeClick}
            >
              proceed to audio analysis
            </SecondaryButton>
          </div>
        </div>
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
          />

          {selectedAnalysisFeature && (
            <div className="flex flex-col w-full lg:w-fit pt-6 space-y-4">
              <div className="text-xl font-semibold text-lightpink mb-1">
                {userAudioData?.file?.name ||
                  userAudioData?.name ||
                  "Input Audio"}
              </div>

              <div className="bg-lightgray/25 rounded-3xl w-full p-4 lg:p-8 pt-10">
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
                    <SecondaryButton
                      onClick={() => handleAnalyzeNewRecording()}
                    >
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
