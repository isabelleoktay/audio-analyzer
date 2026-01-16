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
  { id: "entry", component: EntryQuestions, config: EntryQuestionsConfig },
];

const commonEnd = [
  { id: "final", component: FinalExitSurvey, config: FinalExitConfig },
];

// Shuffle helper
const shuffle = (arr) => [...arr].sort(() => Math.random() - 0.5);

// Build the two condition blocks (control vs tool) in randomized order
const buildConditionBlocks = (taskIndex) => {
  const taskName =
    musaVoiceTestInstructionsConfig[taskIndex]?.task || `Task${taskIndex}`;
  const taskSlug = taskName.replace(/\s+/g, "-").toLowerCase(); // Slugify for ID
  const conditions = [
    { condition: "control", usesTool: false, label: "Control (no tool)" },
    { condition: "tool", usesTool: true, label: "With Tool" },
  ];

  return shuffle(conditions).flatMap((cond) => {
    const sectionKey = `${taskSlug}-${cond.condition}`; // Unique key per section for MongoDB entries
    return [
      {
        id: `record-initial-${sectionKey}`,
        sectionKey: sectionKey,
        component: RecordTask,
        config: musaVoiceTestRecordConfig,
        configIndex: taskIndex,
        metadata: {
          phase: "pre-practice",
          condition: cond.condition,
          usesTool: false,
          taskIndex,
          label: cond.label,
        },
      },
      {
        // Section exit after this condition’s record-practice-record block
        id: `before-practice-survey-${sectionKey}`,
        sectionKey: sectionKey,
        component: SectionSurvey,
        config: SurveyBeforePracticeConfig,
        configIndex: taskIndex,
      },
      {
        id: `practice-${sectionKey}`,
        sectionKey: sectionKey,
        component: Practice,
        config: musaVoiceTestPracticeConfig,
        configIndex: taskIndex,
        metadata: {
          condition: cond.condition,
          usesTool: cond.usesTool,
          taskIndex,
          label: cond.label,
        },
      },
      {
        id: `record-final-${sectionKey}`,
        sectionKey: sectionKey,
        component: RecordTask,
        config: musaVoiceTestRecordConfig,
        configIndex: taskIndex,
        metadata: {
          phase: "post-practice",
          condition: cond.condition,
          usesTool: false,
          taskIndex,
          label: cond.label,
        },
      },
      {
        // Section exit after this condition’s record-practice-record block
        id: `after-practice-survey-${sectionKey}`,
        sectionKey: sectionKey,
        component: SectionSurvey,
        config: SurveyAfterPracticeConfig,
        configIndex: taskIndex,
      },
    ];
  });
};

const taskFlow = (taskIndex) => [
  {
    id: `instructions-${taskIndex}`,
    sectionKey: `instructions-${musaVoiceTestInstructionsConfig[taskIndex]?.task
      .replace(/\s+/g, "-")
      .toLowerCase()}`,
    component: Instructions,
    config: musaVoiceTestInstructionsConfig,
    configIndex: taskIndex,
  },
  ...buildConditionBlocks(taskIndex),
];

export const musaVoiceUserTestFlow = [
  ...commonStart,
  ...taskFlow(0),
  ...commonEnd,
];

export const buildFlowForSelection = (selectedTestFlow) => {
  if (selectedTestFlow === "Full Test Procedure") {
    // Both tasks, random task order; each task internally randomizes control/tool order
    const flows = [taskFlow(0), taskFlow(1)];
    const shuffledTasks = shuffle(flows);
    return [...commonStart, ...shuffledTasks.flat(), ...commonEnd];
  } else if (selectedTestFlow === "Vocal Tone Control") {
    return [...commonStart, ...taskFlow(1), ...commonEnd];
  } else {
    // Default: Pitch Modulation Control
    return [...commonStart, ...taskFlow(0), ...commonEnd];
  }
};
