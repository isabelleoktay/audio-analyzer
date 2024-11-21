import mongoose from "mongoose";

const audioSchema = new mongoose.Schema({
  filename: String,
  mimetype: String,
  size: Number,
  audioBuffer: Buffer,
});

const Audio = mongoose.model("Audio", audioSchema);

export default Audio;
