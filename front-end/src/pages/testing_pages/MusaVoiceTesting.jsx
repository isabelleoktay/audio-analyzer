import { useState } from "react";
import { musaVoiceTestConsentConfig } from "../../data/musaVoiceTestConsentConfig";
import ConsentCardNew from "../../components/testing/ConsentCardNew";
import ProgressBar from "../../components/testing/ProgressBar";

const MusaVoiceTesting = () => {
  const [progressBarIndex, setProgressBarIndex] = useState(0);
  const [progressBarTotalSteps, setProgressBarTotalSteps] = useState(10);
  const [showTaskSelection, setShowTaskSelection] = useState(false);
  const [selectedTestFlow, setSelectedTestFlow] = useState(null);

  const handleClick = (clickTrue) => {
    if (showTaskSelection === false) {
      if (clickTrue) {
        // Proceed to next step in testing
        setShowTaskSelection(true);
      } else {
        // Redirect to home page
        window.location.href = "/";
      }
    } else {
      if (clickTrue) {
        // User selected full test procedure
        setSelectedTestFlow("Full Test Procedure");
      } else {
        // User selected randomly allocated half of procedure
        const testFlows = ["Vocal Tone Control", "Pitch Modulation Control"];
        const randomFlow = testFlows[Math.floor(Math.random() * testFlows.length)];
        setSelectedTestFlow(randomFlow);
      }
      window.location.href = "/musavoice-testing-entryquestions";
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-darkblue">
      <ProgressBar
        currentStep={progressBarIndex}
        totalSteps={progressBarTotalSteps}
      />
      {showTaskSelection ? (
        <div className="flex flex-col items-center justify-center min-h-screen bg-darkblue">
          <ConsentCardNew
            handleClick={handleClick}
            config={musaVoiceTestConsentConfig[1]}
          />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center min-h-screen bg-darkblue">
          <ConsentCardNew
            handleClick={handleClick}
            config={musaVoiceTestConsentConfig[0]}
          />
        </div>
      )}
      ;
    </div>
  );
};

export default MusaVoiceTesting;
