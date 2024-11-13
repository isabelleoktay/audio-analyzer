import axios from "axios";

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL,
});

const processAudio = async (audioFile) => {
  const formData = new FormData();
  formData.append("audioFile", audioFile);

  try {
    const response = await apiClient.post("/api/process-audio", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data; // This will contain the full result from the backend
  } catch (error) {
    console.error("Error processing audio:", error);
    throw error;
  }
};

export { processAudio };
