import { useState, useEffect } from "react";
import Survey from "../components/survey/Survey.jsx";
import musaVoiceSurveyConfig from "../data/musaVoiceSurveyConfig.js";

const MusaVoice = () => {
  const [showIntro, setShowIntro] = useState(true);
  const [showSurvey, setShowSurvey] = useState(true);

  useEffect(() => {
    // Hide intro and show survey after 2 seconds
    const timer = setTimeout(() => setShowIntro(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  const handleSubmit = (answers) => {
    console.log("Survey answers:", answers);
    setShowSurvey(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
      {showIntro ? (
        <h1 className="text-5xl font-bold text-lightpink animate-zoomIn">
          Welcome to MusaVoice!
        </h1>
      ) : showSurvey ? (
        <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
          <Survey config={musaVoiceSurveyConfig} onSubmit={handleSubmit} />
        </div>
      ) : (
        <div className="w-full h-full" />
      )}
    </div>
  );
};

export default MusaVoice;
