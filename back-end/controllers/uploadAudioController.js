import multer from "multer";
import Audio from "../models/Audio.js";

const upload = multer({ storage: multer.memoryStorage() }).single("audioFile");

/**
 * Handles the uploading of an audio file, saving its metadata and buffer to MongoDB.
 *
 * @param {Object} req - The request object.
 * @param {Object} res - The response object.
 * @returns {void}
 */
const uploadAudio = async (req, res) => {
  upload(req, res, async (err) => {
    if (err) {
      return res.status(500).json({ error: "Error uploading file" });
    }

    const audioFile = req.file;
    if (!audioFile) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    try {
      // Save file metadata and buffer to MongoDB
      const audio = new Audio({
        filename: audioFile.originalname,
        mimetype: audioFile.mimetype,
        size: audioFile.size,
        audioBuffer: audioFile.buffer,
      });

      await audio.save();

      res.status(200).json({ message: "File uploaded and saved to database" });
    } catch (error) {
      res.status(500).send("Error saving audio to database");
    }
  });
};

export { uploadAudio };
