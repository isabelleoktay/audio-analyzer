import mongoose from "mongoose";

const emailSchema = new mongoose.Schema({
  email: {
    type: mongoose.Schema.Types.String, // stores email as a string
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

const Email = mongoose.model("Email", emailSchema, "contact-emails");

export default Email;
