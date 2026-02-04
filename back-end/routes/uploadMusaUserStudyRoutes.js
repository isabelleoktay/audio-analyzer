import express from "express";
import {
  uploadMusaUserStudy,
  saveEntrySurvey,
  upsertSection,
  saveSectionField,
  saveExitSurvey,
  //   getStudyBySubject,
} from "../controllers/uploadMusaUserStudyController.js";

const musaUserStudyRouter = express.Router();

musaUserStudyRouter.route("/upload-musa-user-study").post(uploadMusaUserStudy);
musaUserStudyRouter.route("/save-entry-survey").post(saveEntrySurvey);
musaUserStudyRouter.route("/upsert-section").post(upsertSection);
musaUserStudyRouter.route("/save-section-field").post(saveSectionField);
musaUserStudyRouter.route("/save-exit-survey").post(saveExitSurvey);
// musaUserStudyRouter.route("/get-study").get(getStudyBySubject);

export default musaUserStudyRouter;
