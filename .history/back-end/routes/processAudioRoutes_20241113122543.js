import express from "express";
const processAudioRouter = express.Router();
import { processAudio } from "../controllers/audioFeaturesController.js";

processAudioRouter.route("/process-audio").post(processAudio);

export default processAudioRouter;
