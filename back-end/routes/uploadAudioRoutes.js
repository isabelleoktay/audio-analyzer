import express from "express";
const uploadAudioRouter = express.Router();
import { uploadAudio } from "../controllers/uploadAudioController.js";

uploadAudioRouter.route("/upload-audio").post(uploadAudio);

export default uploadAudioRouter;
