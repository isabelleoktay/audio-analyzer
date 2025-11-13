import { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import SurveySection from "../components/survey/SurveySection.jsx";
import { uploadMusaVoiceSessionData } from "../utils/api.js";
import musaVoiceSurveyConfig from "../data/musaVoiceSurveyConfig.js";

const MusaVoice = () => {
  const [showIntro, setShowIntro] = useState(true);
  const [showSurvey, setShowSurvey] = useState(true);
  const [sessionId, setSessionId] = useState(null);
  const [userToken, setUserToken] = useState(null);

  useEffect(() => {
    // Hide intro and show survey after 2 seconds
    const timer = setTimeout(() => setShowIntro(false), 1500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Get or create session ID
    let currentSessionId = sessionStorage.getItem("musaVoiceSessionId");
    if (!currentSessionId) {
      currentSessionId = uuidv4();
      sessionStorage.setItem("musaVoiceSessionId", currentSessionId);
    }
    setSessionId(currentSessionId);

    // Get user token from localStorage
    const token = localStorage.getItem("audio_analyzer_token");
    setUserToken(token);
  }, []);

  const handleSubmit = async (answers) => {
    console.log("Survey answers:", answers);

    try {
      const sessionData = {
        sessionId: sessionId,
        userToken: userToken,
        surveyAnswers: answers,
        timestamp: new Date().toISOString(),
        type: "musaVoice",
      };

      await uploadMusaVoiceSessionData(sessionData);
      console.log("MusaVoice session data uploaded successfully");

      setShowSurvey(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      console.error("Error uploading MusaVoice session data:", error);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
      {showIntro ? (
        <h1 className="text-5xl font-bold text-lightpink animate-zoomIn">
          Welcome to MusaVoice!
        </h1>
      ) : showSurvey ? (
        <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
          <SurveySection
            config={musaVoiceSurveyConfig}
            onSubmit={handleSubmit}
          />
        </div>
      ) : (
        <div className="w-full h-full" />
      )}
    </div>
  );
};

export default MusaVoice;
