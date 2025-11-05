import MusaVoiceSession from "../models/MusaVoiceSession.js";
import dotenv from "dotenv";

dotenv.config();

const uploadMusaVoiceSession = async (req, res) => {
  const { sessionId, userToken, surveyAnswers, timestamp, type } = req.body;

  if (!sessionId || !surveyAnswers) {
    return res.status(400).json({
      error: "Missing required fields: sessionId or surveyAnswers",
    });
  }

  try {
    // Check if session already exists (to prevent duplicates)
    const existingSession = await MusaVoiceSession.findOne({ sessionId });

    if (existingSession) {
      return res.status(409).json({
        error: "Session with this ID already exists",
        sessionId: sessionId,
      });
    }

    // Create a new MusaVoice session document
    const newSession = new MusaVoiceSession({
      sessionId,
      userToken,
      surveyAnswers,
      timestamp: timestamp || new Date(),
      type: type || "musaVoice",
      createdAt: new Date(),
    });

    await newSession.save();

    return res.status(201).json({
      message: "MusaVoice session data uploaded successfully",
      session: {
        id: newSession._id,
        sessionId: newSession.sessionId,
        timestamp: newSession.timestamp,
        type: newSession.type,
      },
    });
  } catch (error) {
    console.error("Error uploading MusaVoice session data:", error);
    res.status(500).json({ error: "Error uploading MusaVoice session data" });
  }
};

export { uploadMusaVoiceSession };
