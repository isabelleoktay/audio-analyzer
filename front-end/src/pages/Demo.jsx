import { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import { uploadMusaVoiceSessionData, cleanupTempFiles } from "../utils/api.js";
import MusaDemoAudioSection from "../components/sections/MusaDemoAudioSection.jsx";
import { AnalysisButtons, SecondaryButton } from "../components/buttons";
import OverlayGraphWithWaveform from "../components/visualizations/OverlayGraphWithWaveform.jsx";
import SimilarityScoreCard from "../components/cards/SimilarityScoreCard";

/**
 * The `Demo` component has option to use pre recorded audios to explore the tool (could do precalculated features as well)
 *
 * @component
 * @param {Object} props - The props passed to the component.
 * @param {boolean} props.uploadsEnabled - Whether uploads are enabled.
 */

const Demo = ({ uploadsEnabled, setUploadsEnabled }) => {
  //   const [showIntro, setShowIntro] = useState(true);
  const [showUploadAudio, setShowUploadAudio] = useState(true);
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
    // Always generate a new sessionId when the page/component mounts
    const newSessionId = uuidv4();
    sessionStorage.setItem("musaVoiceSessionId", newSessionId);
    setSessionId(newSessionId);

    // Get user token from localStorage
    const token = localStorage.getItem("audio_analyzer_token");
    setUserToken(token);
  }, []);

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

  useEffect(() => {
    // output referenceAudioFeatures and inputAudioFeatures as json files for presets saving
    if (
      Object.keys(inputAudioFeatures).length > 0 &&
      Object.keys(referenceAudioFeatures).length > 0
    ) {
      // Create reference features JSON in the correct format
      const referenceJson = {};
      Object.keys(referenceAudioFeatures).forEach((feature) => {
        referenceJson[feature] = {
          data: {
            [selectedModel]: [
              {
                data: referenceAudioFeatures[feature].data || [],
                label: "reference",
              },
            ],
          },
        };
      });

      // Create input features JSON in the correct format
      const inputJson = {};
      Object.keys(inputAudioFeatures).forEach((feature) => {
        inputJson[feature] = {
          data: {
            [selectedModel]: [
              {
                data: inputAudioFeatures[feature].data || [],
                label: "input",
              },
            ],
          },
        };
      });

      // Log to console for verification
      console.log("Reference Features JSON:", referenceJson);
      console.log("Input Features JSON:", inputJson);

      // Optional: Auto-download the files
      const downloadJson = (data, filename) => {
        const jsonString = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonString], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      };

      // Uncomment to auto-download (may be blocked by browser)
      // downloadJson(referenceJson, 'reference-features.json');
      // downloadJson(inputJson, 'input-features.json');

      // Store in localStorage for manual retrieval
      try {
        localStorage.setItem(
          "referenceAudioFeaturesJson",
          JSON.stringify(referenceJson),
        );
        localStorage.setItem(
          "inputAudioFeaturesJson",
          JSON.stringify(inputJson),
        );
        console.log("Features saved to localStorage");
      } catch (e) {
        console.error("Failed to save to localStorage:", e);
      }
    }
  }, [inputAudioFeatures, referenceAudioFeatures, selectedModel]);

  const featureHasModels = ["vocal tone", "pitch mod."].includes(
    selectedAnalysisFeature,
  );

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
      {showUploadAudio ? (
        <MusaDemoAudioSection
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
            inputFileOrBlob={userFileOrBlob}
            referenceFileOrBlob={referenceFileOrBlob}
            inputAudioFeatures={inputAudioFeatures}
            setInputAudioFeatures={setInputAudioFeatures}
            referenceAudioFeatures={referenceAudioFeatures}
            setReferenceAudioFeatures={setReferenceAudioFeatures}
            inputAudioUuid={inputAudioUuid}
            setInputAudioUuid={setInputAudioUuid}
            uploadsEnabled={uploadsEnabled}
            voiceType={selectedVoiceType}
            musaVoiceSessionId={sessionId}
            monitorResources={false}
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
                      inputAudioFeatures[selectedAnalysisFeature]?.data || []
                    }
                    referenceFeatureData={
                      referenceAudioFeatures[selectedAnalysisFeature]?.data ||
                      []
                    }
                    selectedAnalysisFeature={selectedAnalysisFeature}
                    selectedVoiceType={selectedVoiceType}
                    inputAudioDuration={
                      inputAudioFeatures[selectedAnalysisFeature]?.duration
                    }
                    referenceAudioDuration={
                      referenceAudioFeatures[selectedAnalysisFeature]?.duration
                    }
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

export default Demo;
