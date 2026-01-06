import { useState } from "react";
import { SectionExitConfig } from "../../data/musaVoiceTestSurveysConfig";
import SurveySection from "../../components/survey/SurveySection.jsx";

const SectionExitSurvey = ({ currentTask = "Pitch Modulation Control", config = SectionExitConfig}) => {
  const [answers, setAnswers] = useState({});

  const currentTaskConfig = config?.find(
    (task) => task.task === currentTask
  );

  const handleSubmitSectionExitSurvey = async (answers) => {
    console.log("Exit survey answers:", answers);
    setAnswers(answers);
    // Add logic to send entry survey answers to backend and store them

    // Add logic to check if with/without tool test needs to be completed still

    // Add logic to check whether the other task needs to be completed still

    // If all selected tasks are completed, go to exit survey
    window.location.href = "/musavoice-testing-final-survey";
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8 text-center">
        {currentTaskConfig.task}
      </h1>
      <SurveySection
        config={currentTaskConfig.questions}
        onSubmit={handleSubmitSectionExitSurvey}
      />
    </div>
  );
};

export default SectionExitSurvey;
