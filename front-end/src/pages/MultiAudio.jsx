import { useState } from "react";
import UploadAudioCard from "../components/cards/UploadAudioCard";
import SurveyMultiSelect from "../components/survey/SurveyMultiSelect";
import SecondaryButton from "../components/buttons/SecondaryButton";

const VOICE_OPTIONS = [
  "belt",
  "breathy",
  "spoken",
  "vocal fry",
  "inhaled",
  "straight",
  "trill",
  "trillo",
  "vibrato",
];

const MultiAudio = () => {
  const [referenceAudioSource, setReferenceAudioSource] = useState(null);
  const [referenceAudioData, setReferenceAudioData] = useState(null);
  const [userAudioSource, setUserAudioSource] = useState(null);
  const [userAudioData, setUserAudioData] = useState(null);
  const [showError, setShowError] = useState(false);

  const handleAudioSourceChange = (source, type) => {
    console.log(`${type} audio source changed to:`, source);
    if (type === "reference") {
      setReferenceAudioSource(source);
    } else if (type === "user") {
      setUserAudioSource(source);
    }
  };

  const handleAudioDataChange = (audioData, type) => {
    console.log(`${type} audio data updated:`, audioData);
    if (type === "reference") {
      setReferenceAudioData(audioData);
    } else if (type === "user") {
      setUserAudioData(audioData);
    }
    // Now you have access to:
    // - audioData.source: 'upload' or 'recording'
    // - audioData.file: File object (for uploads)
    // - audioData.blob: Blob object (for recordings)
    // - audioData.url: Object URL (for playback)
    // - audioData.name: Recording name
  };

  const getAudioError = (audioSource, audioData) => {
    if (!audioSource) {
      return "Please select upload or record";
    }

    if (audioSource === "upload") {
      if (!audioData?.file) {
        return "No audio file uploaded";
      }
    } else if (audioSource === "record") {
      if (!audioData?.blob) {
        return "No audio recorded";
      }
    }

    return null;
  };

  const referenceError = getAudioError(
    referenceAudioSource,
    referenceAudioData
  );
  const userError = getAudioError(userAudioSource, userAudioData);
  const isFormValid = !referenceError && !userError;

  const handleAnalyzeClick = () => {
    if (!isFormValid) {
      setShowError(true);
      return;
    }
    // Proceed with analysis
    console.log("Analyzing...");
  };

  const getErrorMessage = () => {
    const errors = [];
    if (referenceError) errors.push(`Reference audio: ${referenceError}`);
    if (userError) errors.push(`Latest recording: ${userError}`);
    return errors.length > 0 ? errors.join(" • ") : null;
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-8 w-full gap-4">
      <div className="flex flex-col gap-2 items-center mb-4 h-full w-full">
        <div className="font-bold text-5xl text-electricblue text-center">
          record/upload reference and test recording
        </div>
        <div className="text-lightgray text-xl w-3/4 text-center">
          the system will analyse how closely your test audio matches the vocal
          techniques in the uploaded/recorded reference audio.
        </div>
      </div>
      <div className="flex flex-col gap-2 w-full h-full">
        <UploadAudioCard
          label="reference audio"
          onAudioSourceChange={(source) =>
            handleAudioSourceChange(source, "reference")
          }
          onAudioDataChange={(data) => handleAudioDataChange(data, "reference")}
        />
        <UploadAudioCard
          label="latest recording"
          onAudioSourceChange={(source) =>
            handleAudioSourceChange(source, "user")
          }
          onAudioDataChange={(data) => handleAudioDataChange(data, "user")}
        />
      </div>

      <SurveyMultiSelect
        question="select target technique"
        options={VOICE_OPTIONS}
        columns={3}
        allowOther={false}
      />
      <div className="flex flex-col items-center gap-1">
        <SecondaryButton
          className="h-fit text-xl tracking-widest"
          onClick={handleAnalyzeClick}
        >
          analyze similarity
        </SecondaryButton>
        <div className="h-12 flex items-center">
          {showError && getErrorMessage() && (
            <div className="px-4 py-2 bg-darkpink/20 rounded-3xl text-darkpink text-sm">
              {getErrorMessage()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MultiAudio;
