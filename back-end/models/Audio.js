import mongoose from "mongoose";

/**
 * Defines the schema for the "Audio" collection in MongoDB.
 * Each document represents an audio file with its metadata and associated features.
 */
const audioSchema = new mongoose.Schema({
  _id: { type: String, required: true },
  path: { type: String, required: true }, // all files are stored in /python-service/static/audio
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
