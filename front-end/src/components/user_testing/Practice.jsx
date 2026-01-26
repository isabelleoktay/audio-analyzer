import { useState, useEffect } from "react";
import SecondaryButton from "../../components/buttons/SecondaryButton.jsx";
import HighlightedText from "../text/HighlightedText.jsx";
import AudioPlayButton from "./AudioPlayButton.jsx";
import AudioRecorder from "./AudioRecorder.jsx";
import OverlayGraphWithWaveform from "../visualizations/OverlayGraphWithWaveform.jsx"; // Added import
import {
  processFeatures,
  uploadUserStudySectionField,
} from "../../utils/api.js";
import TertiaryButton from "../buttons/TertiaryButton.jsx";

const Practice = ({
  onNext,
  config,
  configIndex,
  metadata,
  surveyData,
  voiceType = "alto",
}) => {
  // TO DO: need to have voiceType set by the first survey answer !!!
  const baseConfig = config?.[configIndex] ?? {};
  const { condition = "control", usesTool = false, taskIndex } = metadata || {};
  const conditionConfig = baseConfig.conditions?.[condition] ?? {};

  const [hasRecordings, setHasRecordings] = useState(false);
  const [numAnalyses, setNumAnalyses] = useState(0);
  const [inputAudioFeatures, setInputAudioFeatures] = useState({});
  const [selectedModel, setSelectedModel] = useState("CLAP"); // Added for graph toggle
  const [similarityScore, setSimilarityScore] = useState(null); // Added for graph
  const [timeLeft, setTimeLeft] = useState(baseConfig.practiceTime || 420); // Default 7 mins

  const musaVoiceSessionId = surveyData?.sessionId;

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

  const triggerNext = async () => {
    const timeSpent = baseConfig.practiceTime - timeLeft;
    try {
      if (surveyData?.subjectId && metadata?.sectionKey) {
        // Use the existing generic field uploader
        await uploadUserStudySectionField({
          subjectId: surveyData.subjectId,
          sectionKey: metadata.sectionKey,
          field: "practiceData",
          data: { timeSpentSeconds: timeSpent },
          // addEndedAt: true // Optional: if you want to mark section completion time here
        });
      }
    } catch (e) {
      console.error("Failed to save practice data", e);
    }

    onNext({
      lastPracticeCondition: condition,
      lastPracticeUsesTool: usesTool,
      lastPracticeTaskIndex: taskIndex,
    });
  };

  useEffect(() => {
    if (timeLeft <= 0) {
      triggerNext();
      return;
    }

    const timer = setInterval(() => {
      setTimeLeft((prev) => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
  };

  const handleResetAnalysis = () => {
    setHasRecordings(false);
    setInputAudioFeatures({});
    setSimilarityScore(null);
  };

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
      // Create a File object from the blob
      const file = new File([blob], `recording_${Date.now()}.wav`, {
        type: "audio/wav",
      });

      // 1. Process Features
      const featureResult = await processFeatures({
        file, // Pass the File object
        featureLabel,
        voiceType: voiceType,
        useWhisper: selectedModel === "Whisper" || false,
        useCLAP: selectedModel === "CLAP" || true,
        monitorResources: true,
        sessionId: musaVoiceSessionId,
        fileKey: "input",
      });

      // 2. Reorganize and validate data
      const featureHasModels = ["vocal tone", "pitch mod."].includes(
        featureLabel
      );

      let isDataInvalid = false;

      if (!featureHasModels) {
        // Simple features: expect data to be an array
        isDataInvalid = !Array.isArray(featureResult.data);
      } else {
        // For features with models (CLAP/Whisper), at least one model should have data
        const clapData = featureResult.data["CLAP"];
        const whisperData = featureResult.data["Whisper"];

        const hasClapData = Array.isArray(clapData) && clapData.length > 0;
        const hasWhisperData =
          Array.isArray(whisperData) && whisperData.length > 0;

        // Data is invalid only if BOTH models are missing or empty
        isDataInvalid = !hasClapData && !hasWhisperData;
      }

      const featureData = {
        data: isDataInvalid ? "invalid" : featureResult.data,
        sampleRate: featureResult.sample_rate,
        audioUrl: featureResult.audio_url, // Use backend trimmed audio URL
        duration: featureResult.duration || 0,
      };

      setInputAudioFeatures((prev) => ({
        ...prev,
        [featureLabel]: featureData,
      }));
      setNumAnalyses((prev) => prev + 1);
    } catch (error) {
      console.error("Analysis failed:", error);
      // Optional: Reset state on error so the user can try again
      handleResetAnalysis();
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full px-8 mt-16">
      <div className="w-9/12">
        <h2 className="text-4xl text-electricblue font-bold mb-4 text-center">
          Practising {currentTaskConfig.title}
        </h2>
        <p className="text-left text-lg">{currentTaskConfig.instruction}</p>
        <hr className="border-t border-lightgray/20 mb-8 mt-2" />

        <div className="flex flex-col space-y-1">
          <div className="flex justify-between items-end w-full">
            <p className="text-left text-lg text-warmyellow font-semibold">
              Listen and sing the phrase:
            </p>
            <div className="text-electricblue font-mono font-bold bg-electricblue/10 px-3 py-1 rounded-lg border border-electricblue/20">
              {formatTime(timeLeft)}
            </div>
          </div>
          <div className="flex flex-row items-center gap-4 bg-blueblack/50 p-3 rounded-3xl w-full">
            <AudioPlayButton audioUrl="https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3" />
            <HighlightedText
              text={currentTaskConfig.phrase}
              highlightWords={currentTaskConfig.highlightedText}
              highlightClass={currentTaskConfig.highlightClass}
              defaultClass={currentTaskConfig.defaultClass}
              highlightLabel={baseConfig.highlightLabel}
              defaultLabel={baseConfig.defaultLabel}
              highlightLabelColor={baseConfig.highlightLabelColor}
              defaultLabelColor={baseConfig.defaultLabelColor}
              className="text-center justify-center flex-grow"
            />
          </div>
        </div>

        {!currentAnalysis ? (
          <div className="flex flex-col mt-8 space-y-3">
            <div>
              <p className="text-left text-lg font-semibold text-lightpink mb-1">
                Your Recording:
              </p>
              <AudioRecorder
                showAttempts={false}
                analyzeMode={true}
                onAnalyze={handleAnalysis}
                onAttemptsChange={() => {
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
                inputFeatureData={currentAnalysis?.data || []}
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
          <SecondaryButton onClick={triggerNext} disabled={numAnalyses === 0}>
            {numAnalyses > 0
              ? "Continue to the next task."
              : "Please record and analyse at least one attempt to continue"}
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
};

export default Practice;
