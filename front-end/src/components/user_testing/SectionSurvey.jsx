import SurveySection from "../survey/SurveySection.jsx";
import {
  SurveyBeforePracticeConfig,
  SurveyAfterPracticeConfig,
} from "../../config/musaVoiceTestSurveysConfig.js";
import { musaVoiceTestInstructionsConfig } from "../../config/musaVoiceTestInstructionsConfig.js";
import { uploadUserStudySectionField } from "../../utils/api.js";

const SectionSurvey = ({
  onNext,
  config,
  configIndex,
  surveyData = {},
  id,
  sectionKey,
}) => {
  // Derive task name from configIndex, last practice, sectionKey, or selectedTestFlow
  const taskNameFromIndex =
    configIndex !== undefined && typeof configIndex === "number"
      ? musaVoiceTestInstructionsConfig?.[configIndex]?.task
      : undefined;

  const taskNameFromLastPractice =
    surveyData.lastPracticeTaskIndex !== undefined
      ? musaVoiceTestInstructionsConfig?.[surveyData.lastPracticeTaskIndex]
          ?.task
      : undefined;

  // Try to derive from sectionKey if no task name found yet
  const taskNameFromSectionKey =
    !taskNameFromIndex && sectionKey
      ? musaVoiceTestInstructionsConfig?.find(
          (e) => e.sectionKey === sectionKey,
        )?.task
      : undefined;

  const fullTaskName =
    taskNameFromIndex ??
    taskNameFromLastPractice ??
    taskNameFromSectionKey ??
    surveyData.selectedTestFlow ??
    "Pitch Modulation Control";

  // Extract base task name (remove "No Tool" or "Tool" suffix for survey config matching)
  const currentTaskName = fullTaskName.replace(/\s+(No Tool|Tool)$/, "").trim();

  const usesToolFlag = surveyData.lastPracticeUsesTool;

  const currentTaskConfig = config?.find(
    (entry) =>
      entry.task === currentTaskName &&
      (usesToolFlag === undefined ||
        entry.usesTool === undefined ||
        entry.usesTool === usesToolFlag),
  ) ?? { task: "", questions: [] };

  const handleSubmitSurveyBeforePractice = async (submittedAnswers) => {
    try {
      await uploadUserStudySectionField({
        subjectId: surveyData.subjectId,
        sectionKey: sectionKey,
        field: "surveyBeforePracticeAnswers",
        data: submittedAnswers,
        addStartedAt: true,
      });
      // console.log("survey before practice answers uploaded successfully");
      onNext({
        surveyBeforePracticeAnswers: submittedAnswers,
      });
      window.scrollTo(0, 0);
    } catch (error) {
      onNext({
        surveyBeforePracticeAnswers: submittedAnswers,
      });
      console.error("Error uploading survey before practice answers:", error);
    }
  };

  const handleSubmitSurveyAfterPractice = async (submittedAnswers) => {
    try {
      await uploadUserStudySectionField({
        subjectId: surveyData.subjectId,
        sectionKey: sectionKey,
        field: "surveyAfterPracticeAnswers",
        data: submittedAnswers,
        addEndedAt: true,
      });
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
