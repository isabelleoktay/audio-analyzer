import {
  MusaVoiceTesting,
  EntryQuestions,
  Instructions,
  RecordTask,
  Practice,
  SectionSurvey,
  FinalExitSurvey,
} from "../components/user_testing/index.jsx";

import { musaVoiceTestConsentConfig } from "./musaVoiceTestConsentConfig.js";
import { musaVoiceTestInstructionsConfig } from "./musaVoiceTestInstructionsConfig.js";
import { musaVoiceTestRecordConfig } from "./musaVoiceTestRecordConfig.js";
import { musaVoiceTestPracticeConfig } from "./musaVoiceTestPracticeConfig.js";

import {
  EntryQuestionsConfig,
  SurveyBeforePracticeConfig,
  SurveyAfterPracticeConfig,
  FinalExitConfig,
} from "./musaVoiceTestSurveysConfig.js";

const commonStart = [
  {
    id: "consent",
    component: MusaVoiceTesting,
    config: musaVoiceTestConsentConfig,
  },
  //   { id: "entry", component: EntryQuestions, config: EntryQuestionsConfig },
  {
    id: `instructions-intro`,
    sectionKey: `instructions-intro`,
    component: Instructions,
    config: musaVoiceTestInstructionsConfig,
    configIndex: 0,
  },
];

const commonEnd = [
  { id: "final", component: FinalExitSurvey, config: FinalExitConfig },
];

// Shuffle helper
const shuffle = (arr) => [...arr].sort(() => Math.random() - 0.5);

// Build the two condition blocks (control vs tool) in randomized order
const buildConditionBlocks = (taskType) => {
  // taskType should be "pitch" or "vocal"
  const getTaskIndex = (taskType) => (taskType === "pitch" ? 0 : 1);

  const baseTaskLabel =
    taskType === "pitch" ? "Pitch Modulation Control" : "Vocal Tone Control";
  const taskSlug = baseTaskLabel.replace(/\s+/g, "-").toLowerCase(); // Slugify for ID
  const conditions = [
    { condition: "control", usesTool: false, label: "Control (no tool)" },
    { condition: "tool", usesTool: true, label: "With Tool" },
  ];

  return shuffle(conditions).flatMap((cond) => {
    // Find the instruction entry that matches this task and condition (No Tool vs Tool)
    // const condKeyword = cond.condition === "control" ? "No Tool" : "Tool";
    // const instrIndex = musaVoiceTestInstructionsConfig.findIndex(
    //   (e) => e.task?.includes(baseTaskLabel) && e.task?.includes(condKeyword),
    // );

    // const sectionKey = `${taskSlug}-${cond.condition}`;

    const hasNoKeyword = cond.condition === "control";
    const instrIndex = musaVoiceTestInstructionsConfig.findIndex(
      (e) =>
        e.task?.includes(baseTaskLabel) &&
        (hasNoKeyword ? e.task?.includes("No") : !e.task?.includes("No")),
    );

    const sectionKey = `${taskSlug}-${cond.condition}`;

    return [
      {
        id: `instructions-${sectionKey}`,
        sectionKey: sectionKey,
        component: Instructions,
        config: musaVoiceTestInstructionsConfig,
        configIndex: instrIndex,
      },
      {
        id: `record-initial-${sectionKey}`,
        sectionKey: sectionKey,
        component: RecordTask,
        config: musaVoiceTestRecordConfig,
        configIndex: getTaskIndex(taskType),
        metadata: {
          phase: "pre-practice",
          condition: cond.condition,
          usesTool: cond.usesTool,
          taskType,
          label: cond.label,
        },
      },
      {
        // Section exit after this condition’s record-practice-record block
        id: `before-practice-survey-${sectionKey}`,
        sectionKey: sectionKey,
        component: SectionSurvey,
        config: SurveyBeforePracticeConfig,
        configIndex: instrIndex,
      },
      {
        id: `practice-${sectionKey}`,
        sectionKey: sectionKey,
        component: Practice,
        config: musaVoiceTestPracticeConfig,
        configIndex: getTaskIndex(taskType),
        metadata: {
          condition: cond.condition,
          usesTool: cond.usesTool,
          taskType,
          label: cond.label,
        },
      },
      {
        id: `record-final-${sectionKey}`,
        sectionKey: sectionKey,
        component: RecordTask,
        config: musaVoiceTestRecordConfig,
        configIndex: getTaskIndex(taskType),
        metadata: {
          phase: "post-practice",
          condition: cond.condition,
          usesTool: cond.usesTool,
          taskType,
          label: cond.label,
        },
      },
      {
        // Section exit after this condition’s record-practice-record block
        id: `after-practice-survey-${sectionKey}`,
        sectionKey: sectionKey,
        component: SectionSurvey,
        config: SurveyAfterPracticeConfig,
        configIndex: instrIndex,
      },
    ];
  });
};

const taskFlow = (taskType) => [...buildConditionBlocks(taskType)];

export const musaVoiceUserTestFlow = [
  ...commonStart,
  ...taskFlow("pitch"),
  ...commonEnd,
];

export const buildFlowForSelection = (selectedTestFlow) => {
  if (selectedTestFlow === "Full Test Procedure") {
    // Both tasks, random task order; each task internally randomizes control/tool order
    const flows = [taskFlow("pitch"), taskFlow("vocal")];
    const shuffledTasks = shuffle(flows);
    return [...commonStart, ...shuffledTasks.flat(), ...commonEnd];
  } else if (selectedTestFlow === "Vocal Tone Control") {
    return [...commonStart, ...taskFlow("vocal"), ...commonEnd];
  } else {
    // Default: Pitch Modulation Control
    return [...commonStart, ...taskFlow("pitch"), ...commonEnd];
  }
};
