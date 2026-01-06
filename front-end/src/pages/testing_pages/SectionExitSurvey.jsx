import { useState } from "react";
import SurveySection from "../../components/survey/SurveySection.jsx";
import { SectionExitConfig } from "../../data/musaVoiceTestSurveysConfig";

const SectionExitSurvey = ({
  onNext,
  surveyData,
  config = SectionExitConfig,
}) => {
  const [answers, setAnswers] = useState({});

  const currentTask =
    surveyData?.selectedTestFlow ?? "Pitch Modulation Control";

  const currentTaskConfig = config?.find(
    (task) => task.task === currentTask
  ) ?? {
    task: "",
    questions: [],
  };

  const handleSubmitSectionExitSurvey = async (submittedAnswers) => {
    console.log("Exit survey answers:", submittedAnswers);
    setAnswers(submittedAnswers);

    // Add logic to send entry survey answers to backend and store them

    // Add logic to check if with/without tool test needs to be completed still

    // Add logic to check whether the other task needs to be completed still

    // If all selected tasks are completed, go to exit survey
    onNext({
      sectionExitAnswers: submittedAnswers,
    });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full max-w-5xl px-4">
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
