import express from "express";
import { uploadFeedback } from "../controllers/uploadFeedbackController.js";

const feedbackRouter = express.Router();

feedbackRouter.post("/upload-feedback", uploadFeedback);

export default feedbackRouter;
