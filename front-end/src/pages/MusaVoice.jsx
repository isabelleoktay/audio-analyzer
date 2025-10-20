import { useState, useEffect } from "react";
import MusaVoiceStartSurvey from "../components/survey/MusaVoiceStartSurvey";

const MusaVoice = () => {
  const [showIntro, setShowIntro] = useState(true);

  useEffect(() => {
    // Hide intro and show survey after 2 seconds
    const timer = setTimeout(() => setShowIntro(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="flex items-center justify-center min-h-screen">
      {showIntro ? (
        <h1 className="text-5xl font-bold text-lightpink animate-zoomIn">
          Welcome to MusaVoice!
        </h1>
      ) : (
        <MusaVoiceStartSurvey
          onSubmit={(answers) => {
            console.log("Survey submitted:", answers);
          }}
        />
      )}
    </div>
  );
};

export default MusaVoice;
