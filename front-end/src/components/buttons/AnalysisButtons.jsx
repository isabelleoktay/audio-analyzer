// components/AnalysisButtons.jsx
import ButtonGroup from "./ButtonGroup.jsx";
import {
  analysisButtonConfig,
  analysisButtonClassNames,
} from "../../config/analysisButtons.js";

const AnalysisButtons = ({
  selectedInstrument,
  selectedAnalysisFeature,
  onAnalysisFeatureSelect,
}) => {
  if (!analysisButtonConfig[selectedInstrument]) return null;

  const buttons = analysisButtonConfig[selectedInstrument].map((btn) => ({
    ...btn,
    asButton: true,
    onClick: () => onAnalysisFeatureSelect(btn.label),
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
