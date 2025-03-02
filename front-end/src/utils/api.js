import axios from "axios";

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL,
});

/**
 * Processes an audio file by sending it to the backend API.
 *
 * @param {File} audioFile - The audio file to be processed.
 * @param {string} minNote - The minimum note to be considered in the processing.
 * @param {string} maxNote - The maximum note to be considered in the processing.
 * @returns {Promise<Object>} The response data from the backend containing the processing result.
 * @throws {Error} If there is an error during the processing.
 */
const processAudio = async (audioFile, minNote, maxNote) => {
  const formData = new FormData();
  formData.append("audioFile", audioFile);
  formData.append("minNote", minNote);
  formData.append("maxNote", maxNote);

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

/**
 * Uploads an audio file to the server.
 *
 * @param {File} audioFile - The audio file to be uploaded.
 * @returns {Promise<Object>} The response data from the server.
 * @throws Will throw an error if the upload fails.
 */
const uploadAudio = async (audioFile) => {
  const formData = new FormData();
  formData.append("audioFile", audioFile);

  try {
    const response = await apiClient.post("/api/upload-audio", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading audio:", error);
    throw error;
  }
};

export { processAudio, uploadAudio };
