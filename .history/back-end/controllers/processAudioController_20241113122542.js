import multer from "multer";
import axios from "axios";
import FormData from "form-data";

const upload = multer({ storage: multer.memoryStorage() }).single("audioFile");

const processAudio = async (req, res) => {
  upload(req, res, async (err) => {
    if (err) {
      return res.status(500).json({ error: "Error uploading file" });
    }

    const audioFile = req.file;
    if (!audioFile) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const formData = new FormData();
    formData.append("audioFile", audioFile.buffer, audioFile.originalname);

    try {
      const response = await axios.post(
        "http://localhost:8080/process-audio",
        formData,
        {
          headers: formData.getHeaders(),
        }
      );
      res.json(response.data); // Send back the Python service result
    } catch (error) {
      res.status(500).send("Error processing audio");
    }
  });
};

export { processAudio };
