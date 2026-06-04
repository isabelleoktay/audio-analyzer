import Email from "../models/Email.js";
import dotenv from "dotenv";

dotenv.config();

const uploadEmail = async (req, res) => {
  const { email, timestamp } = req.body;

  if (!email) {
    return res.status(400).json({ error: "Missing required field: email" });
  }

  try {
    // Create a new email document
    const newEmail = new Email({
      email,
      timestamp: timestamp || new Date(),
    });

    await newEmail.save();

    return res.status(201).json({
      message: "Email submitted successfully",
      email: newEmail,
      id: newEmail._id,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error uploading email" });
  }
};

export { uploadEmail };
