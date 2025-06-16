import TestSubject from "../models/TestSubject.js";
import dotenv from "dotenv";

dotenv.config();

const uploadTestSubject = async (req, res) => {
  const { subjectId, data } = req.body;

  if (!subjectId || !data) {
    return res
      .status(400)
      .json({ error: "Missing required fields: subjectId or data" });
  }

  try {
    // Check if the subject already exists
    let subject = await TestSubject.findOne({ subjectId });

    if (subject) {
      // Update the existing subject's data
      subject.data = { ...subject.data, ...data }; // Merge existing data with new data
      await subject.save();
      return res
        .status(200)
        .json({ message: "Subject updated successfully", subject });
    } else {
      // Create a new subject
      subject = new TestSubject({ subjectId, data });
      await subject.save();
      return res
        .status(201)
        .json({ message: "Subject created successfully", subject });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error uploading subject data" });
  }
};

export { uploadTestSubject };
