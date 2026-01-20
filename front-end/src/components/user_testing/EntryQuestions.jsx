import { useState } from "react";
import SurveySection from "../survey/SurveySection.jsx";
import { uploadUserStudyEntrySurvey } from "../../utils/api.js";

const EntryQuestions = ({ surveyData, config, onNext }) => {
  const [answers, setAnswers] = useState({});
  const [sectionIndex, setSectionIndex] = useState(0);

  const handleSubmit = (newAnswers) => {
    setAnswers((prev) => ({ ...prev, [sectionIndex]: newAnswers }));
    handleSubmitEntrySurvey({ ...answers, [sectionIndex]: newAnswers });

    window.scrollTo(0, 0);
  };

  const handleSubmitEntrySurvey = async (allAnswers) => {
    try {
      const response = await uploadUserStudyEntrySurvey(
        surveyData.subjectId,
        allAnswers
      );
      // console.log("Entry survey uploaded successfully:", response);
      if (onNext) onNext(); // Advance to next step
    } catch (error) {
      console.error("Error uploading entry survey:", error);
      if (onNext) onNext();
    }
  };

  return (
    <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
      <SurveySection config={config} onSubmit={handleSubmit} />
    </div>
  );
};

export default EntryQuestions;
