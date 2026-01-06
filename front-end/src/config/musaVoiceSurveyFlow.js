import {
  MusaVoiceTesting,
  EntryQuestions,
  Instructions,
  RecordTask,
  Practice,
  SectionExitSurvey,
  FinalExitSurvey,
} from "../pages/testing_pages";

export const musaVoiceSurveyFlow = [
  { id: "consent", component: MusaVoiceTesting },
  { id: "entry", component: EntryQuestions },
  { id: "instructions", component: Instructions },
  { id: "record", component: RecordTask },
  { id: "practice", component: Practice },
  { id: "sectionEnd", component: SectionExitSurvey },
  { id: "final", component: FinalExitSurvey },
];
