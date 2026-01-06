import { EntryQuestionsConfig } from "../../data/musaVoiceTestSurveysConfig.js";
import SurveySection from "../../components/survey/SurveySection.jsx";

const EntryQuestions = ({ onNext }) => {
  const handleSubmitSurvey = async (submittedAnswers) => {
    console.log("Survey answers:", submittedAnswers);
    onNext({
      entrySurveyAnswers: submittedAnswers,
    });
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
