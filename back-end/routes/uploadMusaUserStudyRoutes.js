import express from "express";
import {
  uploadMusaUserStudy,
  saveEntrySurvey,
  upsertSection,
  saveSectionField,
  saveExitSurvey,
  saveEmail,
} from "../controllers/uploadMusaUserStudyController.js";

const musaUserStudyRouter = express.Router();

musaUserStudyRouter.route("/upload-musa-user-study").post(uploadMusaUserStudy);
musaUserStudyRouter.route("/save-entry-survey").post(saveEntrySurvey);
musaUserStudyRouter.route("/upsert-section").post(upsertSection);
musaUserStudyRouter.route("/save-section-field").post(saveSectionField);
musaUserStudyRouter.route("/save-exit-survey").post(saveExitSurvey);
musaUserStudyRouter.route("/save-email").post(saveEmail);

export default musaUserStudyRouter;
