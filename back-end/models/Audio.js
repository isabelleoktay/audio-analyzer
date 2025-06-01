import mongoose from "mongoose";

const audioSchema = new mongoose.Schema({
  filename: { type: String, required: true },
  mimetype: { type: String, required: true },
  size: { type: Number, required: true },
  audioBuffer: { type: Buffer, required: true },
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
