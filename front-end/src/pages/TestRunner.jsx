import { useState } from "react";
import { ThankYou } from "../components/user_testing";
import ProgressBar from "../components/testing/ProgressBar";

function TestRunner({ flow }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const [surveyData, setSurveyData] = useState({});

  const Step = flow[stepIndex].component;
  const stepConfig = flow[stepIndex].config;

  const handleNext = (payload = {}) => {
    // merge payload into surveyData and go to next step (or finish)
    setSurveyData((prev) => ({ ...prev, ...payload }));

    if (stepIndex < flow.length - 1) {
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
      {/* Progress bar container */}
      <div className="fixed w-full">
        <ProgressBar
          currentStep={isFinished ? flow.length : stepIndex} // stepIndex + 1 so it ends at full
          totalSteps={flow.length}
        />
      </div>

      {/* Step content container */}
      <div className="w-full max-w-5xl px-4 flex justify-center">
        {isFinished ? (
          <ThankYou />
        ) : (
          <Step
            onNext={handleNext}
            onPrev={handlePrev}
            stepIndex={stepIndex}
            totalSteps={flow.length}
            config={stepConfig}
            surveyData={surveyData}
          />
        )}
      </div>
    </div>
  );
}

export default TestRunner;
