import mongoose from "mongoose";

const testSubjectSchema = new mongoose.Schema({
  subjectId: { type: String, required: true },
  data: { type: Object, default: {} },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

const TestSubject = mongoose.model("TestSubject", testSubjectSchema);

export default TestSubject;
