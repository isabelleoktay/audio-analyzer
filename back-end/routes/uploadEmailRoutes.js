import express from "express";
import { uploadEmail } from "../controllers/uploadEmailController.js";

const emailRouter = express.Router();

emailRouter.post("/upload-email", uploadEmail);

export default emailRouter;
