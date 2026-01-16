import { useState } from "react";
import SecondaryButton from "../../components/buttons/SecondaryButton.jsx";
import HighlightedText from "../text/HighlightedText.jsx";
import ResponsiveWaveformPlayer from "../visualizations/ResponsiveWaveformPlayer.jsx";
import AudioRecorder from "./AudioRecorder.jsx";
import OverlayGraphWithWaveform from "../visualizations/OverlayGraphWithWaveform.jsx"; // Added import
import { processFeatures } from "../../utils/api.js";
import TertiaryButton from "../buttons/TertiaryButton.jsx";

const Practice = ({ onNext, config, configIndex, metadata, surveyData }) => {
  const [hasRecordings, setHasRecordings] = useState(false);
  const [inputAudioFeatures, setInputAudioFeatures] = useState({});
  const [selectedModel, setSelectedModel] = useState("CLAP"); // Added for graph toggle
  const [similarityScore, setSimilarityScore] = useState(null); // Added for graph

  const { condition = "control", usesTool = false, taskIndex } = metadata || {};
  const musaVoiceSessionId = surveyData?.sessionId;

  const baseConfig = config?.[configIndex] ?? {};
  const conditionConfig = baseConfig.conditions?.[condition] ?? {};

  const currentTaskConfig = {
    title: baseConfig.title,
    instruction: baseConfig.instructions,
    ...conditionConfig,
  };

  const featureLabelMap = {
    "Pitch Modulation Control": "pitch mod.",
    "Vocal Tone Control": "vocal tone",
  };

  const featureLabel = featureLabelMap[baseConfig.task];
  const currentAnalysis = inputAudioFeatures[featureLabel];

  const handleResetAnalysis = () => {
    setHasRecordings(false);
    setInputAudioFeatures({});
    setSimilarityScore(null);
  };

  const formattedGraphData =
    currentAnalysis?.data && currentAnalysis.data !== "invalid"
      ? Object.keys(currentAnalysis.data).map((modelName) => ({
          label: modelName,
          // The API returns an array for each model, so we pass that array directly
          data: currentAnalysis.data[modelName],
        }))
      : currentAnalysis?.data === "invalid"
      ? "invalid"
      : [];

  const handleAnalysis = async (blob) => {
    const featureLabel = featureLabelMap[baseConfig.task];
    if (!featureLabel) return;

    // Immediately switch to analysis view using local blob URL
    const localUrl = URL.createObjectURL(blob);
    setInputAudioFeatures((prev) => ({
      ...prev,
      [featureLabel]: {
        audioUrl: localUrl,
        data: null, // Indicates loading state to the graph
        duration: 0,
      },
    }));

    try {
      // 1. Process Features
      const featureResult = await processFeatures(
        blob,
        featureLabel,
        "voice", // voiceType
        musaVoiceSessionId,
        "input"
      );

      console.log(featureResult);
      // 2. Reorganize and validate data
      const clapData = featureResult.data["CLAP"];
      const whisperData = featureResult.data["Whisper"];

      const isDataInvalid =
        !clapData ||
        !whisperData ||
        !Array.isArray(clapData) ||
        !Array.isArray(whisperData) ||
        clapData.length === 0 ||
        whisperData.length === 0;

      const featureData = {
        data: isDataInvalid ? "invalid" : featureResult.data,
        sampleRate: featureResult.sample_rate,
        audioUrl: featureResult.audio_url || localUrl,
        duration: featureResult.duration || 0,
      };

      setInputAudioFeatures((prev) => ({
        ...prev,
        [featureLabel]: featureData,
      }));
      setHasRecordings(true);

      console.log("Analysis Complete:", featureLabel);
    } catch (error) {
      console.error("Analysis failed:", error);
      // Optional: Reset state on error so the user can try again
      handleResetAnalysis();
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full px-8">
      <div className="w-9/12">
        <h2 className="text-4xl text-electricblue font-bold mb-4 text-center">
          Practising {currentTaskConfig.title}
        </h2>
        <p className="text-left text-lg">{currentTaskConfig.instruction}</p>
        <hr className="border-t border-lightgray/20 mb-8 mt-2" />

        <div className="flex flex-col space-y-1">
          <p className="text-left text-lg text-warmyellow font-semibold">
            The phrase to sing:
          </p>
          <HighlightedText
            text={currentTaskConfig.phrase}
            highlightWords={currentTaskConfig.highlightedText}
            highlightClass={currentTaskConfig.highlightClass}
            defaultClass={currentTaskConfig.defaultClass}
            highlightLabel={baseConfig.highlightLabel}
            defaultLabel={baseConfig.defaultLabel}
            highlightLabelColor={baseConfig.highlightLabelColor}
            defaultLabelColor={baseConfig.defaultLabelColor}
            className="text-center justify-center self-center bg-blueblack/50 p-3 pb-5 rounded-3xl w-full"
          />
        </div>

        {!currentAnalysis ? (
          <div className="flex flex-col mt-8 space-y-3">
            <div>
              <p className="text-left text-lg font-semibold text-lightpink mb-1">
                Reference Audio:
              </p>
              <div className="bg-lightgray/20 p-4 rounded-3xl w-full">
                <ResponsiveWaveformPlayer
                  audioUrl="https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3"
                  highlightedSections={[]}
                  progressColor="#FFD6E8"
                  waveColor="#E0E0E0"
                  showTimeline={false}
                  playButtonColor="text-lightgray"
                />
              </div>
            </div>
            <div>
              <p className="text-left text-lg font-semibold text-lightpink mb-1">
                Your Recording:
              </p>
              <AudioRecorder
                showAttempts={false}
                analyzeMode={true}
                onAnalyze={handleAnalysis}
                onAttemptsChange={() => {
                  setHasRecordings(false);
                  setInputAudioFeatures({});
                }}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col mt-8 space-y-2 w-full">
            <div className="py-4 px-6 bg-lightgray/20 rounded-3xl w-fit self-center">
              <OverlayGraphWithWaveform
                inputAudioURL={currentAnalysis.audioUrl}
                inputFeatureData={formattedGraphData}
                selectedAnalysisFeature={featureLabel}
                inputAudioDuration={currentAnalysis.duration}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                similarityScore={similarityScore}
                setSimilarityScore={setSimilarityScore}
                tooltipMode="none"
              />
            </div>
            <div className="flex justify-end">
              <TertiaryButton onClick={handleResetAnalysis} className="text-sm">
                Analyze another recording
              </TertiaryButton>
            </div>
          </div>
        )}
        <hr className="border-t border-lightgray/20 mb-4 mt-8" />

        <div className="flex justify-center">
          <SecondaryButton
            onClick={() =>
              onNext({
                lastPracticeCondition: condition,
                lastPracticeUsesTool: usesTool,
                lastPracticeTaskIndex: taskIndex,
              })
            }
            disabled={!hasRecordings}
          >
            {hasRecordings
              ? "Continue to the next task."
              : "Please record and analyse at least one attempt to continue"}
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
};

export default Practice;
