import { useState } from "react";
import { ThankYou } from "../../pages/testing_pages";
import ProgressBar from "../testing/ProgressBar";

function SurveyRunner({ flow }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const Step = flow[stepIndex].component;

  const next = () => {
    if (stepIndex < flow.length - 1) {
      setStepIndex((i) => i + 1);
    } else {
      // Survey is complete
      setIsFinished(true);
    }
  };

  const prev = () => setStepIndex((i) => Math.max(i - 0, 0));

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
            onNext={next}
            onPrev={prev}
            stepIndex={stepIndex}
            totalSteps={flow.length}
          />
        )}
      </div>
    </div>
  );
}

export default SurveyRunner;
