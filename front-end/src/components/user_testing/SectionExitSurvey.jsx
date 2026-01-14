import { useState } from "react";
import SurveySection from "../survey/SurveySection.jsx";
import { musaVoiceTestInstructionsConfig } from "../../config/musaVoiceTestInstructionsConfig.js";
import { uploadUserStudySectionEndSurvey } from "../../utils/api.js";

const SectionExitSurvey = ({
  onNext,
  config,
  configIndex,
  surveyData = {},
  sectionId,
}) => {
  // Derive task name from configIndex, otherwise from last practice task index, otherwise from selectedTestFlow
  const taskNameFromIndex =
    configIndex !== undefined
      ? musaVoiceTestInstructionsConfig?.[configIndex]?.task
      : undefined;

  const taskNameFromLastPractice =
    surveyData.lastPracticeTaskIndex !== undefined
      ? musaVoiceTestInstructionsConfig?.[surveyData.lastPracticeTaskIndex]
          ?.task
      : undefined;

  const currentTaskName =
    taskNameFromIndex ??
    taskNameFromLastPractice ??
    surveyData.selectedTestFlow ??
    "Pitch Modulation Control";

  const usesToolFlag = surveyData.lastPracticeUsesTool;

  const currentTaskConfig = config?.find(
    (entry) =>
      entry.task === currentTaskName &&
      (usesToolFlag === undefined ||
        entry.usesTool === undefined ||
        entry.usesTool === usesToolFlag)
  ) ?? { task: "", questions: [] };

  const handleSubmitSectionExitSurvey = async (submittedAnswers) => {
    console.log("Exit survey answers:", submittedAnswers);

    try {
      await uploadUserStudySectionEndSurvey(
        surveyData.subjectId,
        sectionId,
        submittedAnswers
      );
      console.log("Section exit survey uploaded successfully");
    } catch (error) {
      console.error("Error uploading section exit survey:", error);
      // Perhaps still proceed or handle error
    }

    onNext({
      sectionEndSurveyAnswers: submittedAnswers,
    });
    window.scrollTo(0, 0);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full max-w-5xl px-4">
      <h1 className="text-5xl text-electricblue font-bold mt-10 mb-8 text-center">
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
