/**
 * Utility functions for handling preset audio features and precalculated data
 */

import { presetAudios } from "../config/presetAudios.js";

/**
 * Gets the audioFeaturesPath for a preset audio by presetId
 * @param {string} presetId - The preset ID
 * @returns {string|null} The path to precalculated features, or null if not found
 */
export const getPresetAudioFeaturesPath = (presetId) => {
  if (!presetId) return null;
  const preset = presetAudios.find((p) => p.id === presetId);
  return preset?.audioFeaturesPath || null;
};

/**
 * Fetches all precalculated features for a preset
 * @param {string} presetId - The preset ID
 * @returns {Promise<Object|null>} All precalculated features or null if fetch fails
 */
export const fetchPresetFeatures = async (presetId) => {
  const featuresPath = getPresetAudioFeaturesPath(presetId);

  if (!featuresPath) {
    return null;
  }

  try {
    const response = await fetch(featuresPath);
    if (!response.ok) {
      console.warn(
        `Failed to fetch precalculated features from ${featuresPath}: ${response.statusText}`,
      );
      return null;
    }

    const featuresData = await response.json();
    console.log(
      `[Preset Features] Successfully fetched precalculated features for preset: ${presetId}`,
    );
    return featuresData;
  } catch (error) {
    console.warn(
      `Error fetching precalculated features from ${featuresPath}:`,
      error,
    );
    return null;
  }
};
