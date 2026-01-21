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
  },
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
  },
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

const processFeatures = async ({
  file: audioFile,
  featureLabel: feature,
  voiceType,
  useWhisper = true,
  useCLAP = true,
  monitorResources = true,
  sessionId = null,
  fileKey = "input",
} = {}) => {
  const formData = new FormData();
  formData.append("audioFile", audioFile);
  if (sessionId) formData.append("sessionId", sessionId);
  if (fileKey) formData.append("fileKey", fileKey);
  formData.append("monitorResources", String(monitorResources).toLowerCase());

  try {
    if (feature === "dynamics") {
      const response = await pythonClient.post(
        "/python-service/process-dynamics",
        formData,
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
        },
      );
      return response.data;
    } else if (feature === "tempo") {
      const response = await pythonClient.post(
        "/python-service/process-tempo",
        formData,
      );
      return response.data;
    } else if (feature === "vocal tone") {
      formData.append("voiceType", voiceType);
      formData.append("useWhisper", useWhisper);
      formData.append("useCLAP", useCLAP);
      const response = await pythonClient.post(
        "/python-service/process-vocal-tone",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );
      return response.data;
    } else if (feature === "pitch mod.") {
      formData.append("voiceType", voiceType);
      formData.append("useWhisper", useWhisper);
      formData.append("useCLAP", useCLAP);

      const response = await pythonClient.post(
        "/python-service/process-pitch-mod",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );
      return response.data;
    } else if (feature === "vibrato") {
      const response = await pythonClient.post(
        "/python-service/process-vibrato",
        formData,
      );
      return response.data;
    } else if (feature === "phonation") {
      const response = await pythonClient.post(
        "/python-service/process-phonation",
        formData,
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
  feature = null,
) => {
  const formData = new FormData();
  formData.append("file", audioFile);

  //   console.log("group, stage, feature", group, stage, feature);

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
      formData,
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
const uploadAudio = async (
  audioFile,
  id,
  instrument,
  features,
  musaVoiceSessionId = null,
) => {
  try {
    const originalName = audioFile.name;
    const modifiedFileName = `${id}_${originalName}`;
    const modifiedFile = new File([audioFile], modifiedFileName, {
      type: audioFile.type,
    });

    // Upload to Python service (always the same)
    const pythonResponse = await uploadAudioToPythonService(modifiedFile);
    const pythonFilePath = pythonResponse.path;

    // Prepare audioData for MusaVoice session update
    const audioData = {
      audioPath: pythonFilePath,
      instrument,
      features,
      fileName: modifiedFileName,
    };

    if (musaVoiceSessionId) {
      console.log(
        "Uploading to MusaVoice session:",
        musaVoiceSessionId,
        audioData,
      );
      // If MusaVoice session, update the session with audio data
      const backendResponse = await apiClient.post(
        "/api/update-musa-voice-session-audio",
        {
          sessionId: musaVoiceSessionId,
          audioData,
        },
      );
      return backendResponse.data;
    } else {
      // Otherwise, use the generic upload-audio endpoint
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
        },
      );
      return backendResponse.data;
    }
  } catch (error) {
    console.error("Error uploading audio:", error);
    throw error;
  }
};

/**
 * Uploads feedback form answers to the backend.
 *
 * @param {Object} feedbackData - The feedback answers organized by section.
 * @returns {Promise<Object>} The response data from the backend.
 * @throws Will throw an error if the upload fails.
 */
const uploadFeedback = async (feedbackData) => {
  try {
    const response = await apiClient.post("/api/upload-feedback", {
      feedback: feedbackData,
      timestamp: new Date().toISOString(),
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading feedback:", error);
    throw error;
  }
};

/**
 * Uploads MusaVoice session data to the backend.
 *
 * @param {Object} musaVoiceSessionData - The session data including sessionId, userToken, surveyAnswers, etc.
 * @returns {Promise<Object>} The response data from the backend.
 * @throws Will throw an error if the upload fails.
 */
const uploadMusaVoiceSessionData = async (musaVoiceSessionData) => {
  try {
    const response = await apiClient.post("/api/upload-musa-voice-session", {
      ...musaVoiceSessionData, // Spread the data directly, don't wrap in sessionData
      timestamp: musaVoiceSessionData.timestamp || new Date().toISOString(),
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading Musa voice session data:", error);
    throw error;
  }
};

const cleanupTempFiles = async (clearCache = false) => {
  try {
    const response = await pythonClient.post(
      `/python-service/audio/cleanup-temp-files?clear_cache=${clearCache}`,
    );
    console.log("Cleanup response:", response.data);
  } catch (error) {
    console.error("Error cleaning up temporary files:", error);
  }
};

const uploadAllMusaUserStudyData = async (
  subjectId,
  testName,
  entrySurveyAnswers,
  sections,
  exitSurveyAnswers,
) => {
  try {
    const response = await apiClient.post("/api/upload-musa-user-study", {
      subjectId,
      testName,
      entrySurveyAnswers,
      sections,
      exitSurveyAnswers,
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading Musa User Study:", error);
    throw error;
  }
};

const uploadUserStudyEntrySurvey = async (subjectId, entrySurveyAnswers) => {
  try {
    const response = await apiClient.post("/api/save-entry-survey", {
      subjectId,
      entrySurveyAnswers,
    });
    return response.data;
  } catch (error) {
    console.error(
      "Error uploading Musa User Study entry survey answers:",
      error,
    );
    throw error;
  }
};

const uploadUserStudyExitSurvey = async (subjectId, exitSurveyAnswers) => {
  try {
    const response = await apiClient.post("/api/save-exit-survey", {
      subjectId,
      exitSurveyAnswers,
    });
    return response.data;
  } catch (error) {
    console.error(
      "Error uploading Musa User Study exit survey answers:",
      error,
    );
    throw error;
  }
};

const uploadUserStudySectionField = async ({
  subjectId,
  sectionKey,
  field,
  data,
  addStartedAt = false,
  addEndedAt = false,
}) => {
  try {
    const response = await apiClient.post("/api/save-section-field", {
      subjectId,
      sectionKey,
      field,
      data,
      addStartedAt,
      addEndedAt,
    });
    return response.data;
  } catch (error) {
    console.error("Error uploading section field: ", field, error);
    throw error;
  }
};

const upsertUserStudySection = async (subjectId, section) => {
  try {
    const response = await apiClient.post("/api/upsert-section", {
      subjectId,
      section,
    });
    return response.data;
  } catch (error) {
    console.error("Error upserting section of Musa User Study:", error);
    throw error;
  }
};

export {
  uploadAudio,
  processFeatures,
  uploadAudioToPythonService,
  uploadTestSubject,
  cleanupTempFiles,
  startNewSession,
  uploadFeedback,
  uploadMusaVoiceSessionData,
  uploadAllMusaUserStudyData,
  uploadUserStudyEntrySurvey,
  uploadUserStudyExitSurvey,
  uploadUserStudySectionField,
  upsertUserStudySection,
};
