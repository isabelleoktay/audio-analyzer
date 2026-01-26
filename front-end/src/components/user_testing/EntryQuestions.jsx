import { useState } from "react";
import SurveySection from "../survey/SurveySection.jsx";
import { uploadUserStudyEntrySurvey } from "../../utils/api.js";

const EntryQuestions = ({ surveyData, config, onNext }) => {
  const [answers, setAnswers] = useState({});
  const [sectionIndex, setSectionIndex] = useState(0);

  const handleSubmit = (newAnswers) => {
    const updatedAnswers = { ...answers, [sectionIndex]: newAnswers };
    setAnswers(updatedAnswers);

    const flatAnswers = Object.values(updatedAnswers).reduce(
      (acc, curr) => ({ ...acc, ...curr }),
      {}
    );

    handleSubmitEntrySurvey(flatAnswers);

    window.scrollTo(0, 0);
  };

  const handleSubmitEntrySurvey = async (allAnswers) => {
    try {
      console.log(allAnswers);
      await uploadUserStudyEntrySurvey(surveyData.subjectId, allAnswers);
      // console.log("Entry survey uploaded successfully:", response);
    } catch (error) {
      console.error("Error uploading entry survey:", error);
    }

    if (onNext) onNext(allAnswers);
  };

  return (
    <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
      <SurveySection config={config} onSubmit={handleSubmit} />
    </div>
  );
};

export default EntryQuestions;
