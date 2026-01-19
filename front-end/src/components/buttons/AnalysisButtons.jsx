// components/AnalysisButtons.jsx
import ButtonGroup from "./ButtonGroup.jsx";
import {
  analysisButtonConfig,
  analysisButtonClassNames,
} from "../../config/analysisButtons.js";
import { processFeatures, uploadAudio } from "../../utils/api.js";

const AnalysisButtons = ({
  selectedInstrument,
  selectedAnalysisFeature,
  onAnalysisFeatureSelect,
  inputFile,
  referenceFile,
  inputAudioFeatures,
  setInputAudioFeatures,
  referenceAudioFeatures,
  setReferenceAudioFeatures,
  inputAudioUuid,
  setInputAudioUuid,
  uploadsEnabled,
  voiceType,
  musaVoiceSessionId,
  useWhisper = true,
  useCLAP = true,
  monitorResources = false,
}) => {
  //   useEffect(() => {
  //     console.log(
  //       "AnalysisButtons rendered with selectedInstrument:",
  //       selectedInstrument
  //     );
  //   }, [selectedInstrument]);

  if (!analysisButtonConfig[selectedInstrument]) return null;

  // Helper function to process features for a given file
  const fetchAndSetFeatures = async (
    file,
    featureLabel,
    currentFeatures,
    setFeatures,
    useWhisper = true,
    useCLAP = true,
    monitorResources = false,
    sessionId = null,
    fileKey = "input",
  ) => {
    // Only proceed if the feature data doesn't exist yet
    if (!currentFeatures[featureLabel]) {
      // Timing: request sent
      const requestStart = performance.now();

      const featureResult = await processFeatures({
        file,
        featureLabel,
        voiceType,
        useWhisper: useWhisper,
        useCLAP: useCLAP,
        monitorResources: monitorResources,
        sessionId: sessionId,
        fileKey: fileKey,
      });

      // Timing: response received
      const requestEnd = performance.now();
      const duration = (requestEnd - requestStart).toFixed(2);

      if (monitorResources) {
        console.log(
          `[processFeatures] Received response for "${featureLabel}". Duration: ${duration} ms`,
        );
      }
      if (!featureResult || !featureResult.data) {
        console.error(
          `[processFeatures] Invalid response for "${featureLabel}":`,
          featureResult,
        );
        return null;
      }

      let isDataInvalid = false;

      const featureHasModels = ["vocal tone", "pitch mod."].includes(
        featureLabel,
      );

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
        audioUrl: featureResult.audio_url || "",
        duration: featureResult.duration || 0,
      };

      setFeatures((prev) => ({
        ...prev,
        [featureLabel]: featureData,
      }));

      return featureData;
    }
    return currentFeatures[featureLabel];
  };

  const buttons = analysisButtonConfig[selectedInstrument].map((btn) => ({
    ...btn,
    asButton: true,
    onClick: async () => {
      onAnalysisFeatureSelect(btn.label);
      const featureLabel = btn.label;
      let inputFeatureData = null;
      let referenceFeatureData = null;

      // Process Reference File (if it exists)
      if (referenceFile) {
        referenceFeatureData = await fetchAndSetFeatures(
          referenceFile,
          featureLabel,
          referenceAudioFeatures,
          setReferenceAudioFeatures,
          useWhisper, // useWhisper
          useCLAP, // useCLAP
          monitorResources, // monitorResources
          musaVoiceSessionId, // sessionId
          "reference", // fileKey
        );
      }

      // Process Input File (Your Audio)
      if (inputFile) {
        inputFeatureData = await fetchAndSetFeatures(
          inputFile,
          featureLabel,
          inputAudioFeatures,
          setInputAudioFeatures,
          useWhisper, // useWhisper
          useCLAP, // useCLAP
          monitorResources, // monitorResources
          musaVoiceSessionId, // sessionId
          "input", // fileKey
        );
        // console.log("Input feature data:", inputFeatureData);
      }

      // 3. Handle Upload (only for the input file)
      // Upload is only necessary if the input feature data was actually calculated/available,
      // and we use the *newest* feature data (which is either the one we just calculated
      // or the one already in state if we skipped calculation).
      if (uploadsEnabled && inputFile && inputFeatureData) {
        const featuresToUpload = {
          ...inputAudioFeatures,
          [featureLabel]: inputFeatureData,
        };
        const uploadResult = await uploadAudio(
          inputFile,
          inputAudioUuid,
          selectedInstrument,
          featuresToUpload,
          musaVoiceSessionId,
        );
        // console.log(uploadResult);
        setInputAudioUuid(uploadResult.id);
      }
    },
    active: selectedAnalysisFeature === btn.label,
    activeClassName: analysisButtonClassNames.active,
    inactiveClassName: analysisButtonClassNames.inactive,
  }));

  return (
    <div className="w-full flex justify-center">
      <ButtonGroup buttons={buttons} className="justify-center" />
    </div>
  );
};

export default AnalysisButtons;
