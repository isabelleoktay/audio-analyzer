import { useState } from "react";
import SurveySingleSelect from "./SurveySingleSelect";
import SurveyMultiSelect from "./SurveyMultiSelect";
import SurveyScale from "./SurveyScale";
import SurveyMultiScale from "./SurveyMultiScale";

const componentMap = {
    singleselect: SurveySingleSelect,
    multiselect: SurveyMultiSelect,
    scale: SurveyScale,
    multiscale: SurveyMultiScale,
};

const Survey = ({ config, onSubmit }) => {
  const [answers, setAnswers] = useState({});

  const handleAnswerChange = (question, answer) => {
    setAnswers((prev) => ({ ...prev, [question]: answer }));
  };

  return (
    <div className="flex flex-col gap-8">
      {config.map((item, index) => {
        const Component = componentMap[item.type];
        return (
          <Component
            key={index}
            {...item}
            onChange={(val) => handleAnswerChange(item.question, val)}
          />
        );
      })}

      <div className="flex justify-center mt-6">
        <button
          onClick={() => onSubmit(answers)}
          className="bg-darkpink text-white px-6 py-2 rounded-xl hover:bg-lightpink"
        >
          Submit
        </button>
      </div>
    </div>
  );
};

export default Survey;
