import express from "express";
import {
  uploadMusaVoiceSession,
  updateMusaVoiceSessionAudio,
} from "../controllers/uploadMusaVoiceSessionController.js";

const musaVoiceSessionRouter = express.Router();

musaVoiceSessionRouter.post(
  "/upload-musa-voice-session",
  uploadMusaVoiceSession
);

musaVoiceSessionRouter.post(
  "/update-musa-voice-session-audio",
  updateMusaVoiceSessionAudio
);

export default musaVoiceSessionRouter;
