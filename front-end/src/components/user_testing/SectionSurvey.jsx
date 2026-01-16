import SurveySection from "../survey/SurveySection.jsx";
import {
  SurveyBeforePracticeConfig,
  SurveyAfterPracticeConfig,
} from "../../config/musaVoiceTestSurveysConfig.js";
import { musaVoiceTestInstructionsConfig } from "../../config/musaVoiceTestInstructionsConfig.js";
import {
  uploadUserStudySurveyAfterPractice,
  uploadUserStudySurveyBeforePractice,
} from "../../utils/api.js";

const SectionSurvey = ({
  onNext,
  config,
  configIndex,
  surveyData = {},
  id,
  sectionKey,
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

  console.log("SectionSurvey currentTaskName:", currentTaskName);

  const usesToolFlag = surveyData.lastPracticeUsesTool;

  const currentTaskConfig = config?.find(
    (entry) =>
      entry.task === currentTaskName &&
      (usesToolFlag === undefined ||
        entry.usesTool === undefined ||
        entry.usesTool === usesToolFlag)
  ) ?? { task: "", questions: [] };

  const handleSubmitSurveyBeforePractice = async (submittedAnswers) => {
    try {
      await uploadUserStudySurveyBeforePractice(
        surveyData.subjectId,
        sectionKey,
        submittedAnswers
      );
      // console.log("survey before practice answers uploaded successfully");
      onNext({
        surveyBeforePracticeAnswers: submittedAnswers,
      });
      window.scrollTo(0, 0);
    } catch (error) {
      console.error("Error uploading survey before practice answers:", error);
    }
  };

  const handleSubmitSurveyAfterPractice = async (submittedAnswers) => {
    try {
      await uploadUserStudySurveyAfterPractice(
        surveyData.subjectId,
        sectionKey,
        submittedAnswers
      );
      // console.log("survey after practice answers uploaded successfully");
      onNext({
        surveyAfterPracticeAnswers: submittedAnswers,
      });
      window.scrollTo(0, 0);
    } catch (error) {
      console.error("Error uploading survey after practice answers:", error);
    }
  };

  const handleSubmit = async (submittedAnswers) => {
    if (config === SurveyBeforePracticeConfig) {
      await handleSubmitSurveyBeforePractice(submittedAnswers);
    } else if (config === SurveyAfterPracticeConfig) {
      await handleSubmitSurveyAfterPractice(submittedAnswers);
    } else {
      console.warn("Unknown survey config; no action taken on submit.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full max-w-5xl px-4">
      <h1 className="text-5xl text-electricblue font-bold mt-10 mb-8 text-center">
        {currentTaskConfig.task}
      </h1>

      <SurveySection
        config={currentTaskConfig.questions}
        onSubmit={handleSubmit}
      />
    </div>
  );
};

export default SectionSurvey;
