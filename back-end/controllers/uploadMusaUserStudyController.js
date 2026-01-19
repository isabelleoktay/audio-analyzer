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
        if (!section.sectionKey) continue; // skip invalid

        const updated = await MusaUserTest.findOneAndUpdate(
          { subjectId, "sections.sectionKey": section.sectionKey },
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

// Upsert a single section (by sectionKey) for a subject
const upsertSection = async (req, res) => {
  const { subjectId } = req.params;
  const section = req.body;

  if (!subjectId || !section || !section.sectionKey)
    return res
      .status(400)
      .json({ ok: false, error: "Missing subjectId or section.sectionKey" });

  try {
    let doc = await MusaUserTest.findOneAndUpdate(
      { subjectId, "sections.sectionKey": section.sectionKey },
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

/**
 * Generic function to save/update a section field
 * @param {Object} req.body
 *  - subjectId: string
 *  - sectionKey: string
 *  - field: string (name of the field to set, e.g., "surveyBeforePracticeAnswers")
 *  - data: any (value to set)
 *  - addEndedAt: boolean (whether to also update endedAt timestamp)
 */
const saveSectionField = async (req, res) => {
  const {
    subjectId,
    sectionKey,
    field,
    data,
    addStartedAt,
    addEndedAt = false,
  } = req.body;

  if (!subjectId || !sectionKey || !field || data === undefined) {
    return res.status(400).json({
      ok: false,
      error: "Missing body params (subjectId, sectionKey, field, data)",
    });
  }

  try {
    // Build the $set object dynamically
    const setObj = { [`sections.$.${field}`]: data };

    if (addStartedAt) {
      setObj["sections.$.startedAt"] = new Date();
    }

    if (addEndedAt) {
      setObj["sections.$.endedAt"] = new Date();
    }

    // Try updating existing section
    let doc = await MusaUserTest.findOneAndUpdate(
      { subjectId, "sections.sectionKey": sectionKey },
      { $set: setObj },
      { new: true }
    );

    // If section doesn't exist, push a new one
    if (!doc) {
      const newSection = { sectionKey, [field]: data };
      if (addEndedAt) newSection.endedAt = new Date();
      if (addStartedAt) newSection.startedAt = new Date();

      doc = await MusaUserTest.findOneAndUpdate(
        { subjectId },
        { $push: { sections: newSection } },
        { new: true }
      );
    }

    if (!doc) {
      return res.status(404).json({ ok: false, error: "Subject not found" });
    }

    return res.status(200).json({ ok: true, doc });
  } catch (err) {
    console.error(err);
    return res
      .status(500)
      .json({ ok: false, error: "Failed to save section field" });
  }
};

// Save exit survey and optionally mark completed
const saveExitSurvey = async (req, res) => {
  const { subjectId, exitSurveyAnswers } = req.body;
  const { markCompleted } = req.query;

  if (!subjectId || !exitSurveyAnswers)
    return res
      .status(400)
      .json({ ok: false, error: "Missing subjectId or answers" });

  try {
    const doc = await MusaUserTest.updateExitSurveyAnswers(
      subjectId,
      exitSurveyAnswers,
      {
        markCompleted: markCompleted === "true",
      }
    );
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
  saveSectionField,
  saveExitSurvey,
  //   getStudyBySubject,
};
