import MusaUserTest from "../models/MusaUserTest.js";
import dotenv from "dotenv";

dotenv.config();

// Generic create-or-update endpoint for a user's study data
// Accepts: { subjectId, testName?, entrySurveyAnswers?, sections?, exitSurveyAnswers? }
const uploadMusaUserStudy = async (req, res) => {
  const {
    subjectId,
    testName,
    entrySurveyAnswers,
    sections,
    exitSurveyAnswers,
  } = req.body;

  if (!subjectId) {
    return res
      .status(400)
      .json({ ok: false, error: "Missing required field: subjectId" });
  }

  if (!testName && !entrySurveyAnswers && !sections && !exitSurveyAnswers) {
    return res.status(400).json({ ok: false, error: "Nothing to save" });
  }

  try {
    let doc = await MusaUserTest.findOne({ subjectId });

    if (!doc) {
      // create new
      const newDoc = new MusaUserTest({
        subjectId,
        testName,
        entrySurveyAnswers,
        sections,
        exitSurveyAnswers,
      });
      await newDoc.save();
      return res
        .status(201)
        .json({ ok: true, message: "Study created", doc: newDoc });
    }

    // update existing doc: selectively set provided fields
    const setObj = {};
    if (testName) setObj.testName = testName;
    if (entrySurveyAnswers) setObj.entrySurveyAnswers = entrySurveyAnswers;
    if (exitSurveyAnswers) setObj.exitSurveyAnswers = exitSurveyAnswers;

    if (Object.keys(setObj).length > 0) {
      doc = await MusaUserTest.findOneAndUpdate(
        { subjectId },
        { $set: setObj },
        { new: true }
      );
    }

    // handle sections upsert individually (if sections array provided)
    if (Array.isArray(sections) && sections.length) {
      for (const section of sections) {
        if (!section.sectionId) continue; // skip invalid

        const updated = await MusaUserTest.findOneAndUpdate(
          { subjectId, "sections.sectionId": section.sectionId },
          { $set: { "sections.$": section } },
          { new: true }
        );

        if (!updated) {
          // push as new section
          doc = await MusaUserTest.findOneAndUpdate(
            { subjectId },
            { $push: { sections: section } },
            { new: true }
          );
        } else {
          doc = updated;
        }
      }
    }

    return res.status(200).json({ ok: true, message: "Study updated", doc });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ ok: false, error: "Failed to save study" });
  }
};

// Save or replace entry survey answers
const saveEntrySurvey = async (req, res) => {
  // console.log("saveEntrySurvey req.body:", req.body);
  const { subjectId, entrySurveyAnswers } = req.body;
  // console.log(
  //   "subjectId:",
  //   subjectId,
  //   "entrySurveyAnswers:",
  //   !!entrySurveyAnswers
  // );

  if (!subjectId || !entrySurveyAnswers)
    return res
      .status(400)
      .json({ ok: false, error: "Missing subjectId or entrySurveyAnswers" });

  try {
    const doc = await MusaUserTest.updateEntrySurveyAnswers(
      subjectId,
      entrySurveyAnswers
    );
    return res.status(200).json({ ok: true, doc });
  } catch (err) {
    console.error(err);
    return res
      .status(500)
      .json({ ok: false, error: "Failed to save entry survey" });
  }
};

// Upsert a single section (by sectionId) for a subject
const upsertSection = async (req, res) => {
  const { subjectId } = req.params;
  const section = req.body;

  if (!subjectId || !section || !section.sectionId)
    return res
      .status(400)
      .json({ ok: false, error: "Missing subjectId or section.sectionId" });

  try {
    let doc = await MusaUserTest.findOneAndUpdate(
      { subjectId, "sections.sectionId": section.sectionId },
      { $set: { "sections.$": section } },
      { new: true }
    );

    if (!doc) {
      doc = await MusaUserTest.findOneAndUpdate(
        { subjectId },
        { $push: { sections: section } },
        { new: true, upsert: true }
      );
    }

    return res.status(200).json({ ok: true, doc });
  } catch (err) {
    console.error(err);
    return res
      .status(500)
      .json({ ok: false, error: "Failed to upsert section" });
  }
};

// Update only the section end survey answers for a given sectionId
const saveSectionEndSurvey = async (req, res) => {
  const { subjectId, sectionId } = req.params;
  const answers = req.body;

  if (!subjectId || !sectionId || !answers)
    return res
      .status(400)
      .json({ ok: false, error: "Missing path params or body" });

  try {
    const doc = await MusaUserTest.findOneAndUpdate(
      { subjectId, "sections.sectionId": sectionId },
      {
        $set: {
          "sections.$.sectionEndSurveyAnswers": answers,
          "sections.$.endedAt": new Date(),
        },
      },
      { new: true }
    );

    if (!doc)
      return res
        .status(404)
        .json({ ok: false, error: "Section or subject not found" });
    return res.status(200).json({ ok: true, doc });
  } catch (err) {
    console.error(err);
    return res
      .status(500)
      .json({ ok: false, error: "Failed to save section end survey" });
  }
};

// Save exit survey and optionally mark completed
const saveExitSurvey = async (req, res) => {
  const { subjectId } = req.params;
  const answers = req.body;
  const { markCompleted } = req.query;

  if (!subjectId || !answers)
    return res
      .status(400)
      .json({ ok: false, error: "Missing subjectId or answers" });

  try {
    const doc = await MusaUserTest.updateExitSurveyAnswers(subjectId, answers, {
      markCompleted: markCompleted === "true",
    });
    return res.status(200).json({ ok: true, doc });
  } catch (err) {
    console.error(err);
    return res
      .status(500)
      .json({ ok: false, error: "Failed to save exit survey" });
  }
};

// // Get study by subjectId
// const getStudyBySubject = async (req, res) => {
//   const { subjectId } = req.params;
//   if (!subjectId)
//     return res.status(400).json({ ok: false, error: "Missing subjectId" });

//   try {
//     const doc = await MusaUserTest.findOne({ subjectId });
//     if (!doc) return res.status(404).json({ ok: false, error: "Not found" });
//     return res.status(200).json({ ok: true, doc });
//   } catch (err) {
//     console.error(err);
//     return res.status(500).json({ ok: false, error: "Failed to fetch study" });
//   }
// };

export {
  uploadMusaUserStudy,
  saveEntrySurvey,
  upsertSection,
  saveSectionEndSurvey,
  saveExitSurvey,
  //   getStudyBySubject,
};
