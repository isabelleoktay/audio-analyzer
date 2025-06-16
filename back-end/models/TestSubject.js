import mongoose from "mongoose";

/**
 * Defines the schema for the "TestSubject" collection in MongoDB.
 * Each document represents a test subject with associated data and metadata.
 */
const testSubjectSchema = new mongoose.Schema({
  subjectId: { type: String, required: true },
  data: { type: Object, default: {} }, // holds all subject data collected, pointing to audio file paths and features
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

const TestSubject = mongoose.model("TestSubject", testSubjectSchema);

export default TestSubject;
