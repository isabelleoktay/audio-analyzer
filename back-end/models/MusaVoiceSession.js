import mongoose from "mongoose";

const musaVoiceSessionSchema = new mongoose.Schema({
  sessionId: {
    type: String,
    required: true,
  },
  userToken: {
    type: String,
    required: false,
  },
  surveyAnswers: {
    type: mongoose.Schema.Types.Mixed,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
  type: {
    type: String,
    default: "musaVoice",
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

const MusaVoiceSession = mongoose.model(
  "MusaVoiceSession",
  musaVoiceSessionSchema
);

export default MusaVoiceSession;
