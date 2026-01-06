import { useState } from "react";
import { musaVoiceTestConsentConfig } from "../../data/musaVoiceTestConsentConfig";
import ConsentCardNew from "../../components/testing/ConsentCardNew";

const MusaVoiceTesting = ({ onNext }) => {
  const [showTaskSelection, setShowTaskSelection] = useState(false);

  const handleClick = (clickTrue) => {
    if (!showTaskSelection) {
      if (clickTrue) {
        setShowTaskSelection(true);
      } else {
        window.location.href = "/";
      }
    } else {
      let flow;

      if (clickTrue) {
        flow = "Full Test Procedure";
      } else {
        const testFlows = ["Vocal Tone Control", "Pitch Modulation Control"];
        flow = testFlows[Math.floor(Math.random() * testFlows.length)];
      }

      onNext({
        selectedTestFlow: flow,
      });
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-darkblue">
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
