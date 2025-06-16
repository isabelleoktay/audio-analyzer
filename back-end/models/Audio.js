import mongoose from "mongoose";

const audioSchema = new mongoose.Schema({
  path: { type: String, required: true },
  instrument: { type: String, required: true },
  features: {
    type: Object,
    default: {},
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

const Audio = mongoose.model("Audio", audioSchema);

export default Audio;
