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

    const { id, features, instrument, audioPath } = req.body;

    if (!id) {
      return res.status(400).json({ error: "ID is required" });
    }

    try {
      // Check if an audio entry with the given ID exists
      let audio = await Audio.findById(id);

      if (audio) {
        // Update the existing audio entry
        if (audioPath) {
          audio.path = audioPath;
        }

        if (features) {
          audio.features = {
            ...audio.features,
            ...JSON.parse(features),
          };
        }

        if (instrument) {
          audio.instrument = instrument;
        }

        await audio.save();

        return res.status(200).json({
          message: "Audio entry updated successfully",
          id: audio._id,
        });
      } else {
        // Create a new audio entry if it doesn't exist
        audio = new Audio({
          _id: id, // Use the provided ID for the new entry
          path: audioPath || "unknown",
          instrument: instrument || "unknown",
          features: features ? JSON.parse(features) : {},
        });

        await audio.save();

        return res.status(201).json({
          message: "New audio entry created successfully",
          id: audio._id,
        });
      }
    } catch (error) {
      console.error("Error processing audio:", error);
      res.status(500).send("Error processing audio");
    }
  });
};

export { uploadAudio };
