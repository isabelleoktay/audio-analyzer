import axios from "axios";
import { tokenManager } from "./tokenManager.js";

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL,
});

const pythonClient = axios.create({
  baseURL: process.env.REACT_APP_PYTHON_SERVICE_BASE_URL,
});

pythonClient.interceptors.request.use(
  async (config) => {
    try {
      const token = await tokenManager.ensureValidToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    } catch (error) {
      console.error("Error getting token for request:", error);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

pythonClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    if (error.response?.status === 401) {
      console.log("Token expired, generating new token and retrying...");
      tokenManager.clearToken();

      try {
        await tokenManager.generateToken();
        // Retry the original request with new token
        const token = await tokenManager.getToken();
        error.config.headers.Authorization = `Bearer ${token}`;
        return pythonClient.request(error.config);
      } catch (retryError) {
        console.error("Error retrying request with new token:", retryError);
        return Promise.reject(error);
      }
    }
    return Promise.reject(error);
  }
);

const startNewSession = async () => {
  try {
    tokenManager.clearToken();
    await tokenManager.generateToken();
    console.log("New session started with fresh token");
  } catch (error) {
    console.error("Error starting new session:", error);
    throw error;
  }
};

const processFeatures = async (audioFile, feature) => {
  const formData = new FormData();
  formData.append("audioFile", audioFile);

  try {
    if (feature === "dynamics") {
      const response = await pythonClient.post(
        "/python-service/process-dynamics",
        formData
      );
      return response.data;
    } else if (feature === "pitch") {
      formData.append("minNote", "F3");
      formData.append("maxNote", "B6");

      const response = await pythonClient.post(
        "/python-service/process-pitch",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return response.data;
    } else if (feature === "tempo") {
      const response = await pythonClient.post(
        "/python-service/process-tempo",
        formData
      );
      return response.data;
    } else if (feature === "vibrato") {
      const response = await pythonClient.post(
        "/python-service/process-vibrato",
        formData
      );
      return response.data;
    } else if (feature === "phonation") {
      const response = await pythonClient.post(
        "/python-service/process-phonation",
        formData
      );
      return response.data;
    } else {
      return [];
    }
  } catch (error) {
    console.error("Error processing data:", error);
    throw error;
  }
};

/**
 * Uploads an audio file to the Python backend.
 *
 * @param {File} audioFile - The audio file to be uploaded.
 * @param {string} [group] - The group for the audio file (e.g., "feedback", "none").
 * @param {string} [stage] - The stage for the audio file (e.g., "before", "during", "after").
 * @returns {Promise<Object>} The response data from the Python backend.
 * @throws Will throw an error if the upload fails.
 */
const uploadAudioToPythonService = async (
  audioFile,
  group = null,
  stage = null,
  feature = null
) => {
  const formData = new FormData();
  formData.append("file", audioFile);

  console.log("group, stage, feature", group, stage, feature);

  if (group) {
    formData.append("group", group);
  }

  if (stage) {
    formData.append("stage", stage);
  }

  if (feature) {
    console.log("Feature to upload:", feature);
    formData.append("feature", feature);
  }

  try {
    const response = await pythonClient.post(
      "/python-service/audio/upload",
      formData
    );
    return response.data;
  } catch (error) {
    console.error("Error uploading audio to Python backend:", error);
    throw error;
  }
};

/**
 * Uploads a test subject to the backend.
 *
 * @param {string} subjectId - The unique ID of the test subject.
 * @param {Object} data - The data associated with the test subject.
 * @returns {Promise<Object>} The response data from the backend.
 * @throws Will throw an error if the upload fails.
 */
const uploadTestSubject = async (subjectId, data) => {
  try {
    const response = await apiClient.post("/api/upload-test-subject", {
      subjectId,
      data,
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading test subject:", error);
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
const uploadAudio = async (audioFile, id, instrument, features) => {
  try {
    const originalName = audioFile.name;
    const modifiedFileName = `${id}_${originalName}`;
    const modifiedFile = new File([audioFile], modifiedFileName, {
      type: audioFile.type,
    });

    const pythonResponse = await uploadAudioToPythonService(modifiedFile);
    const pythonFilePath = pythonResponse.path;

    const formData = new FormData();
    formData.append("audioPath", pythonFilePath);
    formData.append("instrument", instrument);

    if (id) {
      formData.append("id", id);
    }

    formData.append("features", JSON.stringify(features));

    const backendResponse = await apiClient.post(
      "/api/upload-audio",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    return backendResponse.data;
  } catch (error) {
    console.error("Error uploading audio:", error);
    throw error;
  }
};

const cleanupTempFiles = async () => {
  try {
    const response = await pythonClient.post(
      "/python-service/audio/cleanup-temp-files"
    );
    console.log("Cleanup response:", response.data);
  } catch (error) {
    console.error("Error cleaning up temporary files:", error);
  }
};

export {
  uploadAudio,
  processFeatures,
  uploadAudioToPythonService,
  uploadTestSubject,
  cleanupTempFiles,
  startNewSession,
};
