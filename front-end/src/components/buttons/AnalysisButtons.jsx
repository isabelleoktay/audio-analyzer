// components/AnalysisButtons.jsx
import ButtonGroup from "./ButtonGroup.jsx";
import {
  analysisButtonConfig,
  analysisButtonClassNames,
} from "../../config/analysisButtons.js";
import { processFeatures } from "../../utils/api.js";

const AnalysisButtons = ({
  selectedInstrument,
  selectedAnalysisFeature,
  onAnalysisFeatureSelect,
  audioFile,
  audioFeatures,
  setAudioFeatures,
  sampleRate,
  setSampleRate,
}) => {
  if (!analysisButtonConfig[selectedInstrument]) return null;

  const buttons = analysisButtonConfig[selectedInstrument].map((btn) => ({
    ...btn,
    asButton: true,
    onClick: async () => {
      onAnalysisFeatureSelect(btn.label);
      if (!audioFeatures[`${btn.label}`]) {
        const result = await processFeatures(audioFile, btn.label);
        if (!sampleRate) setSampleRate(result.sample_rate);
        setAudioFeatures({
          ...audioFeatures,
          [btn.label]: result[`${btn.label}`],
        });
        console.log(result);
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
