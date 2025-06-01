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
    const { id, features, instrument } = req.body; // Extract `id` and `features` from the request body

    if (!audioFile && !id) {
      return res
        .status(400)
        .json({ error: "No file uploaded or audio ID provided" });
    }

    try {
      if (id) {
        // Update an existing audio entry
        const existingAudio = await Audio.findById(id);

        if (!existingAudio) {
          return res.status(404).json({ error: "Audio entry not found" });
        }

        // Update features and optionally replace the audio buffer if a new file is uploaded
        if (audioFile) {
          existingAudio.filename = audioFile.originalname;
          existingAudio.mimetype = audioFile.mimetype;
          existingAudio.size = audioFile.size;
          existingAudio.audioBuffer = audioFile.buffer;
        }

        if (features) {
          existingAudio.features = {
            ...existingAudio.features,
            ...JSON.parse(features),
          };
        }

        await existingAudio.save();

        return res.status(200).json({
          message: "Audio entry updated successfully",
          id: existingAudio._id,
        });
      } else {
        // Create a new audio entry
        const audio = new Audio({
          filename: audioFile.originalname,
          mimetype: audioFile.mimetype,
          size: audioFile.size,
          audioBuffer: audioFile.buffer,
          instrument: instrument || "unknown",
          features: features ? JSON.parse(features) : {},
        });

        await audio.save();

        console.log("Audio saved:", audio);

        return res.status(200).json({
          message: "File uploaded and saved to database",
          id: audio._id,
        });
      }
    } catch (error) {
      console.error("Error saving audio:", error);
      res.status(500).send("Error saving audio to database");
    }
  });
};

export { uploadAudio };
