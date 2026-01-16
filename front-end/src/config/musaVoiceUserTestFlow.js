import {
  MusaVoiceTesting,
  EntryQuestions,
  Instructions,
  RecordTask,
  Practice,
  SectionExitSurvey,
  FinalExitSurvey,
} from "../components/user_testing/index.jsx";

import { musaVoiceTestConsentConfig } from "./musaVoiceTestConsentConfig.js";
import { musaVoiceTestInstructionsConfig } from "./musaVoiceTestInstructionsConfig.js";
import { musaVoiceTestRecordConfig } from "./musaVoiceTestRecordConfig.js";
import { musaVoiceTestPracticeConfig } from "./musaVoiceTestPracticeConfig.js";

import {
  EntryQuestionsConfig,
  SectionExitConfig,
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
  const conditions = [
    { condition: "control", usesTool: false, label: "Control (no tool)" },
    { condition: "tool", usesTool: true, label: "With Tool" },
  ];

  return shuffle(conditions).flatMap((cond) => {
    const suffix = `${taskIndex}-${cond.condition}`;
    return [
      {
        id: `record-initial-${suffix}`,
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
        id: `practice-${suffix}`,
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
        id: `record-final-${suffix}`,
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
        id: `sectionEnd-${suffix}`,
        component: SectionExitSurvey,
        config: SectionExitConfig,
        configIndex: taskIndex,
      },
    ];
  });
};

const taskFlow = (taskIndex) => [
  {
    id: `instructions-${taskIndex}`,
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
