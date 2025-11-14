import express from "express";
import { uploadMusaVoiceSession } from "../controllers/uploadMusaVoiceSessionController.js";

const musaVoiceSessionRouter = express.Router();

musaVoiceSessionRouter.post(
  "/upload-musa-voice-session",
  uploadMusaVoiceSession
);

export default musaVoiceSessionRouter;
