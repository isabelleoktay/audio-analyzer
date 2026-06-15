import { useState, useEffect, useRef, useEffectEvent } from "react";
import { v4 as uuidv4 } from "uuid";
import { uploadMusaVoiceSessionData, cleanupTempFiles } from "../utils/api.js";
import {
  fetchPresetFeatures,
  getPresetAudioPath,
  isPresetAudio,
} from "../utils/presetFeaturesUtils.js";
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
  const previousFeaturesJsonRef = useRef({ input: "", reference: "" });

  useEffect(() => {
    // Always generate a new sessionId when the page/component mounts
    const newSessionId = uuidv4();
    sessionStorage.setItem("ICMEdemoSessionId", newSessionId);
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
    setSelectedVoiceType(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleAnalysisFeatureSelect = (feature) => {
    setSimilarityScore(null);
    setSelectedAnalysisFeature(feature);
  };

  // useEffect(() => {
  //   if (
  //     Object.keys(inputAudioFeatures).length > 0 &&
  //     Object.keys(referenceAudioFeatures).length > 0
  //   ) {
  //     const inputJson = JSON.stringify(inputAudioFeatures);
  //     const referenceJson = JSON.stringify(referenceAudioFeatures);

  //     if (
  //       previousFeaturesJsonRef.current.input === inputJson &&
  //       previousFeaturesJsonRef.current.reference === referenceJson
  //     ) {
  //       return;
  //     }

  //     previousFeaturesJsonRef.current = {
  //       input: inputJson,
  //       reference: referenceJson,
  //     };

  //     const downloadJson = (data, filename) => {
  //       const jsonString = JSON.stringify(data, null, 2);
  //       const blob = new Blob([jsonString], { type: "application/json" });
  //       const url = URL.createObjectURL(blob);
  //       const link = document.createElement("a");
  //       link.href = url;
  //       link.download = filename;
  //       document.body.appendChild(link);
  //       link.click();
  //       document.body.removeChild(link);
  //       URL.revokeObjectURL(url);
  //     };

  //     // const refName = referenceAudioData?.file?.name || referenceAudioData?.name || "Input Audio"
  //     // const inputName = userAudioData?.file?.name || userAudioData?.name || "Input Audio"

  //     // downloadJson(referenceAudioFeatures, `${refName}.json`);
  //     // setTimeout(
  //     //   () => downloadJson(inputAudioFeatures,  `${inputName}.json`),
  //     //   300,
  //     // );

  //     try {
  //       localStorage.setItem("referenceAudioFeaturesJson", referenceJson);
  //       localStorage.setItem("inputAudioFeaturesJson", inputJson);
  //       console.log("Features saved to localStorage");
  //     } catch (e) {
  //       console.error("Failed to save to localStorage:", e);
  //     }
  //   }
  // }, [inputAudioFeatures, referenceAudioFeatures]);

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

  /**
   * Fetches precalculated features for preset audios and pre-populates feature state
   * If a preset has an audioFeaturesPath defined, loads the precalculated features
   * instead of requiring model inference. Falls back gracefully if not available.
   */
  const getPresetAudioFeatures = async (userAudioData, referenceAudioData) => {
    const inputFeatures = {};
    const referenceFeatures = {};

    // Fetch precalculated features for input audio if it's a preset
    if (userAudioData?.presetId) {
      const presetFeatures = await fetchPresetFeatures(userAudioData.presetId);
      if (presetFeatures) {
        // Get the correct local preset audio path as fallback
        const presetAudioPath = getPresetAudioPath(userAudioData.presetId);

        // Preset JSON already has the correct structure matching the API response format
        for (const [featureLabel, featureData] of Object.entries(
          presetFeatures,
        )) {
          // Use the local preset audio path, falling back to the URL from preset features if needed
          inputFeatures[featureLabel] = {
            ...featureData,
            audioUrl: presetAudioPath || featureData.audioUrl,
          };
        }
        console.log(
          `[Demo] Loaded precalculated features for input preset: ${userAudioData.name}`,
        );
      }
    }

    // Fetch precalculated features for reference audio if it's a preset
    if (referenceAudioData?.presetId) {
      const presetFeatures = await fetchPresetFeatures(
        referenceAudioData.presetId,
      );
      if (presetFeatures) {
        // Get the correct local preset audio path as fallback
        const presetAudioPath = getPresetAudioPath(referenceAudioData.presetId);

        // Preset JSON already has the correct structure matching the API response format
        for (const [featureLabel, featureData] of Object.entries(
          presetFeatures,
        )) {
          // Use the local preset audio path, falling back to the URL from preset features if needed
          referenceFeatures[featureLabel] = {
            ...featureData,
            audioUrl: presetAudioPath || featureData.audioUrl,
          };
        }
        console.log(
          `[Demo] Loaded precalculated features for reference preset: ${referenceAudioData.name}`,
        );
      }
    }

    return { inputFeatures, referenceFeatures };
  };

  const userFileOrBlob = getAudioFileOrBlob(userAudioData);
  const referenceFileOrBlob = getAudioFileOrBlob(referenceAudioData);

  return (
    <div className="flex items-center justify-center min-h-screen">
      {showUploadAudio ? (
        <MusaDemoAudioSection
          onProceed={async ({
            userAudioData,
            referenceAudioData,
            referenceVoiceType,
            selectedTechniques,
            userAudioSource,
            referenceAudioSource,
          }) => {
            // Your logic to proceed to analysis
            cleanupTempFiles(true);
            setShowUploadAudio(false);
            setUserAudioData(userAudioData);
            setReferenceAudioData(referenceAudioData);
            setSelectedVoiceType(referenceVoiceType);
            setSelectedTechniques(selectedTechniques);
            setUserAudioSource(userAudioSource);
            setReferenceAudioSource(referenceAudioSource);

            // Attempt to load precalculated features for presets
            const { inputFeatures, referenceFeatures } =
              await getPresetAudioFeatures(userAudioData, referenceAudioData);

            // Pre-populate feature state if precalculated features are available
            if (Object.keys(inputFeatures).length > 0) {
              setInputAudioFeatures(inputFeatures);
            }
            if (Object.keys(referenceFeatures).length > 0) {
              setReferenceAudioFeatures(referenceFeatures);
            }

            setAnalyzeAudio(true);
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
            uploadsEnabled={
              isPresetAudio(userAudioData) ? false : uploadsEnabled
            }
            voiceType={selectedVoiceType}
            monitorResources={false}
          />

          {selectedAnalysisFeature && (
            <div className="flex flex-col w-full lg:w-fit pt-6 space-y-1">
              <div className="text-xl font-semibold text-lightpink">
                {referenceAudioData?.file?.name ||
                  referenceAudioData?.name ||
                  "Reference Audio"}{" "}
                VS{" "}
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
                    inputAudioName={
                      userAudioData?.file?.name ||
                      userAudioData?.name ||
                      "Input Audio"
                    }
                    referenceAudioName={
                      referenceAudioData?.file?.name ||
                      referenceAudioData?.name ||
                      "Reference Audio"
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
