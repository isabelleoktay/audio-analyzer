import mongoose from "mongoose";

const feedbackSchema = new mongoose.Schema({
  feedback: {
    type: mongoose.Schema.Types.Mixed, // Stores the nested feedback object
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

const Feedback = mongoose.model("Feedback", feedbackSchema, "feedback-forms");

export default Feedback;
