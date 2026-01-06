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
import {
  EntryQuestionsConfig,
  SectionExitConfig,
  FinalExitConfig,
} from "./musaVoiceTestSurveysConfig.js";

export const musaVoiceUserTestFlow = [
  {
    id: "consent",
    component: MusaVoiceTesting,
    config: musaVoiceTestConsentConfig,
  },
  { id: "entry", component: EntryQuestions, config: EntryQuestionsConfig },
  {
    id: "instructions",
    component: Instructions,
    config: musaVoiceTestInstructionsConfig,
  },
  { id: "record", component: RecordTask },
  { id: "practice", component: Practice },
  { id: "sectionEnd", component: SectionExitSurvey, config: SectionExitConfig },
  { id: "final", component: FinalExitSurvey, config: FinalExitConfig },
];
