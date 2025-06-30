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
  uploadedFile,
  audioFeatures,
  setAudioFeatures,
  audioUuid,
  setAudioUuid,
  uploadsEnabled,
}) => {
  if (!analysisButtonConfig[selectedInstrument]) return null;

  const buttons = analysisButtonConfig[selectedInstrument].map((btn) => ({
    ...btn,
    asButton: true,
    onClick: async () => {
      onAnalysisFeatureSelect(btn.label);
      if (!audioFeatures[btn.label]) {
        const featureResult = await processFeatures(uploadedFile, btn.label);

        const isDataInvalid = !Array.isArray(featureResult.data);

        const featureData = {
          data: isDataInvalid ? "invalid" : featureResult.data, // Set to "invalid" if all values are NaN
          sampleRate: featureResult.sample_rate,
          audioUrl: featureResult.audio_url || "",
          duration: featureResult.duration || 0,
        };

        setAudioFeatures((prev) => ({
          ...prev,
          [btn.label]: featureData,
        }));

        // console.log(featureResult);

        if (uploadsEnabled) {
          // console.log("Uploading audio...");
          const uploadResult = await uploadAudio(
            uploadedFile,
            audioUuid,
            selectedInstrument,
            { ...audioFeatures, [btn.label]: featureData }
          );
          // console.log(uploadResult);
          setAudioUuid(uploadResult.id);
        }
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
