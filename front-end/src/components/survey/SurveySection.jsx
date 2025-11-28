import { useState, useCallback } from "react";
import SurveySingleSelect from "./SurveySingleSelect";
import SurveyMultiScale from "./SurveyMultiScale";
import SurveyTextAnswer from "./SurveyTextAnswer";
import SurveyStatementRating from "./SurveyStatementRating";
import SecondaryButton from "../buttons/SecondaryButton";
import MultiSelectCard from "../cards/MultiSelectCard.jsx";

const componentMap = {
  singleselect: SurveySingleSelect,
  multiselect: MultiSelectCard,
  multiscale: SurveyMultiScale,
  textAnswer: SurveyTextAnswer,
  statementRating: SurveyStatementRating,
};

const SurveySection = ({
  config,
  onSubmit,
  sectionTitle,
  buttonText,
  backButtonClick,
  savedAnswers = {},
}) => {
  const [answers, setAnswers] = useState(savedAnswers || {});

  const handleAnswerChange = useCallback((question, answer) => {
    setAnswers((prev) => ({ ...prev, [question]: answer }));
  }, []);

  const handleSubmit = () => {
    // console.log(answers);
    onSubmit(answers);
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Section heading */}
      {sectionTitle && (
        <h2 className="text-xl font-bold text-lightgray text-center">
          {sectionTitle}
        </h2>
      )}

      {config.map((item) => {
        const Component = componentMap[item.type];
        if (!Component) return null; // skip unknown types
        return (
          <Component
            key={item.question}
            {...item}
            onChange={(val) => handleAnswerChange(item.question, val)}
            value={answers[item.question]}
          />
        );
      })}

      <div className="flex justify-center mt-6 gap-6">
        {backButtonClick && (
          <SecondaryButton
            onClick={() => backButtonClick(answers)}
            className="ml-4 bg-lightpink/50 hover:bg-lightpink/70"
          >
            Back
          </SecondaryButton>
        )}
        <SecondaryButton onClick={handleSubmit}>
          {buttonText || "Submit"}
        </SecondaryButton>
      </div>
    </div>
  );
};

export default SurveySection;
