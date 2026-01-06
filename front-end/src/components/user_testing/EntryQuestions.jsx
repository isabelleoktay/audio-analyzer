import SurveySection from "../survey/SurveySection.jsx";

const EntryQuestions = ({ onNext, config }) => {
  const handleSubmitSurvey = async (submittedAnswers) => {
    console.log("Survey answers:", submittedAnswers);
    onNext({
      entrySurveyAnswers: submittedAnswers,
    });
    window.scrollTo(0, 0);
  };

  return (
    <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
      <SurveySection config={config} onSubmit={handleSubmitSurvey} />
    </div>
  );
};

export default EntryQuestions;
