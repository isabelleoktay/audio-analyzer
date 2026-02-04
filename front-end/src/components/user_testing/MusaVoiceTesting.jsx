import { useState } from "react";
import ConsentCardNew from "../testing/ConsentCardNew";

const MusaVoiceTesting = ({ onNext, config }) => {
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

      console.log("Selected test flow:", flow);

      onNext({
        selectedTestFlow: flow,
      });
      window.scrollTo(0, 0);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-darkblue scroll-top">
      {showTaskSelection ? (
        <div className="flex flex-col items-center justify-center min-h-screen bg-darkblue">
          <ConsentCardNew handleClick={handleClick} config={config[1]} />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center min-h-screen bg-darkblue">
          <ConsentCardNew handleClick={handleClick} config={config[0]} />
        </div>
      )}
      ;
    </div>
  );
};

export default MusaVoiceTesting;
