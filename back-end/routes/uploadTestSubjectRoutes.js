import express from "express";
const uploadTestSubjectRouter = express.Router();
import { uploadTestSubject } from "../controllers/uploadTestSubjectController.js";

uploadTestSubjectRouter.route("/upload-test-subject").post(uploadTestSubject);

export default uploadTestSubjectRouter;
