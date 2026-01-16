import mongoose from "mongoose";

const SectionSchema = new mongoose.Schema({
  sectionId: { type: String, required: true },
  startedAt: Date,
  endedAt: Date,
  instructionConfidence: { type: String },
  recordingBeforePractice: {
    audioId: String,
    analysis: mongoose.Schema.Types.Mixed,
  },
  surveyBeforePractice: { type: mongoose.Schema.Types.Mixed, required: true },
  practiceData: { type: mongoose.Schema.Types.Mixed, required: true },
  recordingAfterPractice: {
    audioId: String,
    analysis: mongoose.Schema.Types.Mixed,
  },
  sectionEndSurveyAnswers: {
    type: mongoose.Schema.Types.Mixed,
    required: true,
  },
});

const musaUserTestSchema = new mongoose.Schema({
  testName: { type: String }, // e.g. VTC User Study, VTE User Study, etc.
  subjectId: { type: String, required: true },
  entrySurveyAnswers: { type: mongoose.Schema.Types.Mixed },
  sections: { type: [SectionSchema], default: [] },
  exitSurveyAnswers: { type: mongoose.Schema.Types.Mixed },
  timestamp: { type: Date, default: Date.now },
});

/**
 * Upsert a section by `sectionName`.
 * If a section with the same `sectionName` exists for the session, it will be replaced.
 * Otherwise the section will be pushed onto the `sections` array.
 */

musaUserTestSchema.statics.upsertSectionByName = async function (
  sessionId,
  sectionName,
  sectionDoc
) {
  const existing = await this.findOneAndUpdate(
    { sessionId, "sections.sectionName": sectionName },
    { $set: { "sections.$": sectionDoc } },
    { new: true }
  );
  if (existing) return existing;
  return this.findOneAndUpdate(
    { sessionId },
    { $push: { sections: sectionDoc } },
    { new: true, upsert: true }
  );
};

/** Update arbitrary fields on a section (keeps other fields intact) */

musaUserTestSchema.statics.updateSectionFieldsByName = function (
  sessionId,
  sectionName,
  fields
) {
  const setObj = {};
  for (const key of Object.keys(fields)) {
    setObj[`sections.$.${key}`] = fields[key];
  }
  return this.findOneAndUpdate(
    { sessionId, "sections.sectionName": sectionName },
    { $set: setObj },
    { new: true }
  );
};

/** Update only the section end survey answers */

musaUserTestSchema.statics.updateSectionEndSurveyByName = function (
  sessionId,
  sectionName,
  answers
) {
  return this.findOneAndUpdate(
    { sessionId, "sections.sectionName": sectionName },
    {
      $set: {
        "sections.$.sectionEndSurveyAnswers": answers,
        "sections.$.endedAt": new Date(),
      },
    },
    { new: true }
  );
};

/** Update only the entry survey answers for a subject */

musaUserTestSchema.statics.updateEntrySurveyAnswers = function (
  subjectId,
  answers
) {
  return this.findOneAndUpdate(
    { subjectId },
    { $set: { entrySurveyAnswers: answers } },
    { new: true, upsert: true }
  );
};

/** Update only the final exit survey answers (optionally mark completed) */

musaUserTestSchema.statics.updateExitSurveyAnswers = function (
  subjectId,
  answers,
  opts = { markCompleted: false }
) {
  const setObj = { exitSurveyAnswers: answers };
  if (opts.markCompleted) setObj.completed = true;
  return this.findOneAndUpdate(
    { subjectId },
    { $set: { exitSurveyAnswers: answers } },
    { new: true, upsert: true }
  );
};

const MusaUserTest = mongoose.model(
  "MusaUserTest",
  musaUserTestSchema,
  "musa-user-test"
);

export default MusaUserTest;
