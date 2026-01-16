import { useState } from "react";
import { ThankYou } from "../components/user_testing";
import ProgressBar from "../components/testing/ProgressBar";
import { buildFlowForSelection } from "../config/musaVoiceUserTestFlow.js";

function TestRunner({ flow }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const [surveyData, setSurveyData] = useState({
    subjectId: `subject_${Date.now()}`,
  });
  const [currentFlow, setCurrentFlow] = useState(flow);

  const Step = currentFlow[stepIndex].component;
  const stepConfig = currentFlow[stepIndex].config;
  const configIndex = currentFlow[stepIndex].configIndex;
  const metadata = currentFlow[stepIndex].metadata;

  const handleNext = (payload = {}) => {
    // If payload contains selectedTestFlow, build the appropriate flow
    if (payload.selectedTestFlow) {
      const newFlow = buildFlowForSelection(payload.selectedTestFlow);
      setCurrentFlow(newFlow);
    }

    setSurveyData((prev) => ({ ...prev, ...payload }));

    if (stepIndex < currentFlow.length - 1) {
      setStepIndex((i) => i + 1);
    } else {
      setIsFinished(true);
    }
  };

  const handlePrev = (payload = {}) => {
    setSurveyData((prev) => ({ ...prev, ...payload }));
    setStepIndex((i) => Math.max(0, i - 1));
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center bg-darkblue">
      <div className="fixed w-full">
        <ProgressBar
          currentStep={isFinished ? currentFlow.length : stepIndex}
          totalSteps={currentFlow.length}
        />
      </div>

      <div className="w-full max-w-5xl px-4 flex justify-center">
        {isFinished ? (
          <ThankYou />
        ) : (
          <Step
            onNext={handleNext}
            onPrev={handlePrev}
            stepIndex={stepIndex}
            totalSteps={currentFlow.length}
            config={stepConfig}
            configIndex={configIndex}
            metadata={metadata}
            surveyData={surveyData}
            id={currentFlow[stepIndex].id}
            sectionKey={currentFlow[stepIndex].sectionKey}
          />
        )}
      </div>
    </div>
  );
}

export default TestRunner;
