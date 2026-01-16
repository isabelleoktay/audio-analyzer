import express from "express";
import {
  uploadMusaUserStudy,
  saveEntrySurvey,
  upsertSection,
  saveSurveyAfterPractice,
  saveExitSurvey,
  saveSurveyBeforePractice,
  //   getStudyBySubject,
} from "../controllers/uploadMusaUserStudyController.js";

const musaUserStudyRouter = express.Router();

musaUserStudyRouter.route("/upload-musa-user-study").post(uploadMusaUserStudy);
musaUserStudyRouter.route("/save-entry-survey").post(saveEntrySurvey);
musaUserStudyRouter.route("/upsert-section").post(upsertSection);
musaUserStudyRouter
  .route("/save-survey-before-practice")
  .post(saveSurveyBeforePractice);
musaUserStudyRouter
  .route("/save-survey-after-practice")
  .post(saveSurveyAfterPractice);
musaUserStudyRouter.route("/save-exit-survey").post(saveExitSurvey);
// musaUserStudyRouter.route("/get-study").get(getStudyBySubject);

export default musaUserStudyRouter;
