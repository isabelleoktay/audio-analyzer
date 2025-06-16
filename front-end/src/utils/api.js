import axios from "axios";
import { v4 as uuidv4 } from "uuid";

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL,
});

const pythonClient = axios.create({
  baseURL: "http://localhost:8080",
});

const processFeatures = async (audioFile, feature) => {
  const formData = new FormData();
  formData.append("audioFile", audioFile);

  try {
    if (feature === "dynamics") {
      const response = await pythonClient.post(
        "/python-service/process-dynamics",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
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
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return response.data;
    } else if (feature === "vibrato") {
      const response = await pythonClient.post(
        "/python-service/process-vibrato",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return response.data;
    } else if (feature === "phonation") {
      const response = await pythonClient.post(
        "/python-service/process-phonation",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      return response.data;
    } else {
      return [];
    }
  } catch (error) {
    console.error("Error processing dynamics:", error);
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
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
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
    const uuid = uuidv4();
    const originalName = audioFile.name;
    const modifiedFileName = `${uuid}_${originalName}`;
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
// const uploadAudio = async (audioFile, id, instrument, features) => {
//   const formData = new FormData();
//   formData.append("audioFile", audioFile);
//   formData.append("instrument", instrument);

//   if (id) {
//     formData.append("id", id);
//   }

//   formData.append("features", JSON.stringify(features));

//   try {
//     const response = await apiClient.post("/api/upload-audio", formData, {
//       headers: {
//         "Content-Type": "multipart/form-data",
//       },
//     });
//     return response.data;
//   } catch (error) {
//     console.error("Error uploading audio:", error);
//     throw error;
//   }
// };

export {
  uploadAudio,
  processFeatures,
  uploadAudioToPythonService,
  uploadTestSubject,
};
