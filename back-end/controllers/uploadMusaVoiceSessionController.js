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

const updateMusaVoiceSessionAudio = async (req, res) => {
  const { sessionId, audioData } = req.body;

  if (!sessionId || !audioData) {
    return res.status(400).json({
      error: "Missing required fields: sessionId or audioData",
    });
  }

  try {
    // Find the session by sessionId
    const session = await MusaVoiceSession.findOne({ sessionId });

    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }

    // Attach or update audioData field (add this field to your schema if needed)
    session.audioData = {
      ...session.audioData,
      ...audioData,
    };

    await session.save();

    return res.status(200).json({
      message: "Audio data updated for MusaVoice session",
      session: {
        id: session._id,
        sessionId: session.sessionId,
        audioData: session.audioData,
      },
    });
  } catch (error) {
    console.error("Error updating MusaVoice session audio data:", error);
    res
      .status(500)
      .json({ error: "Error updating MusaVoice session audio data" });
  }
};

export { uploadMusaVoiceSession, updateMusaVoiceSessionAudio };
