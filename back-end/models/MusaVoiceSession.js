import mongoose from "mongoose";

const musaVoiceSessionSchema = new mongoose.Schema({
  sessionId: {
    type: String,
    required: true,
    unique: true,
  },
  userToken: {
    type: String,
    required: false, // Optional since user might not have a token
  },
  surveyAnswers: {
    type: mongoose.Schema.Types.Mixed, // Stores the nested survey answers object
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
