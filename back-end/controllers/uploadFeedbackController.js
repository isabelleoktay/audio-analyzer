import Feedback from "../models/Feedback.js";
import dotenv from "dotenv";

dotenv.config();

const uploadFeedback = async (req, res) => {
  const { feedback, timestamp } = req.body;

  if (!feedback) {
    return res.status(400).json({ error: "Missing required field: feedback" });
  }

  try {
    // Create a new feedback document
    const newFeedback = new Feedback({
      feedback,
      timestamp: timestamp || new Date(),
    });

    await newFeedback.save();

    return res.status(201).json({
      message: "Feedback submitted successfully",
      feedback: newFeedback,
      id: newFeedback._id,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error uploading feedback" });
  }
};

export { uploadFeedback };
