import { useState } from "react";
import { EntryQuestionsConfig } from "../../data/musaVoiceTestSurveysConfig.js";
import SurveySection from "../../components/survey/SurveySection.jsx";

const EntryQuestions = () => {
  const [answers, setAnswers] = useState({});

  const handleSubmitSurvey = async (answers) => {
    console.log("Survey answers:", answers);
    setAnswers(answers);
    // Add logic to send entry survey answers to backend and store them
    window.location.href = "/musavoice-testing-instructions";
  };

  return (
    <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
      <SurveySection
        config={EntryQuestionsConfig}
        onSubmit={handleSubmitSurvey}
      />
    </div>
  );
};

export default EntryQuestions;
