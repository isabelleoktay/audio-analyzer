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
        console.log("AUDIO FILE");
        console.log(uploadedFile);
        const featureResult = await processFeatures(uploadedFile, btn.label);
        const featureData = {
          data: featureResult[btn.label], // this must match the key in `result`
          xAxis: featureResult.x_axis || [],
          sampleRate: featureResult.sample_rate,
          audioUrl: featureResult.audio_url || "",
          highlightedDataSection:
            featureResult?.highlighted_section?.frame || {},
          highlightedAudioSection:
            { ...featureResult?.highlighted_section?.sample } || {},
        };

        setAudioFeatures((prev) => ({
          ...prev,
          [btn.label]: featureData,
        }));

        console.log(featureResult);

        if (uploadsEnabled) {
          console.log("Uploading audio...");
          const uploadResult = await uploadAudio(
            uploadedFile,
            audioUuid,
            selectedInstrument,
            { ...audioFeatures, [btn.label]: featureData }
          );
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
