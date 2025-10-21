import { useState } from "react";
import SurveySingleSelect from "./SurveySingleSelect";
import SurveyMultiSelect from "./SurveyMultiSelect";
import SurveyMultiScale from "./SurveyMultiScale";
import SurveyTextAnswer from "./SurveyTextAnswer";
import SurveyStatementRating from "./SurveyStatementRating";
import SecondaryButton from "../buttons/SecondaryButton";

const componentMap = {
  singleselect: SurveySingleSelect,
  multiselect: SurveyMultiSelect,
  multiscale: SurveyMultiScale,
  textAnswer: SurveyTextAnswer,
  statementRating: SurveyStatementRating,
};

const Survey = ({
  config,
  onSubmit,
  sectionTitle,
  buttonText,
  backButtonClick,
}) => {
  const [answers, setAnswers] = useState({});

  const handleAnswerChange = (question, answer) => {
    setAnswers((prev) => ({ ...prev, [question]: answer }));
  };

  return (
    <div className="flex flex-col gap-8">
      {/* Section heading */}
      {sectionTitle && (
        <h2 className="text-xl font-bold text-lightgray text-center">
          {sectionTitle}
        </h2>
      )}

      {config.map((item, index) => {
        const Component = componentMap[item.type];
        if (!Component) return null; // skip unknown types
        return (
          <Component
            key={index}
            {...item}
            onChange={(val) => handleAnswerChange(item.question, val)}
          />
        );
      })}

      <div className="flex justify-center mt-6 gap-6">
        {backButtonClick && (
          <SecondaryButton
            onClick={backButtonClick}
            className="ml-4 bg-lightpink/50 hover:bg-lightpink/70"
          >
            Back
          </SecondaryButton>
        )}
        <SecondaryButton onClick={onSubmit}>
          {buttonText || "Submit"}
        </SecondaryButton>
      </div>
    </div>
  );
};

export default Survey;
