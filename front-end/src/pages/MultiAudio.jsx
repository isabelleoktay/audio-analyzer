import { useState } from "react";
import UploadAudioCard from "../components/cards/UploadAudioCard";
import MultiSelectCard from "../components/cards/MultiSelectCard";
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

  const handleAudioSourceChange = (source, type) => {
    // console.log(`${type} audio source changed to:`, source);
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
  };

  // Check if audio is ready for analysis
  const isAudioReady = (audioSource, audioData) => {
    if (!audioSource) return false;

    if (audioSource === "upload") {
      return audioData?.file !== null && audioData?.file !== undefined;
    } else if (audioSource === "record") {
      return audioData?.blob !== null && audioData?.blob !== undefined;
    }

    return false;
  };

  const isReferenceReady = isAudioReady(
    referenceAudioSource,
    referenceAudioData
  );
  const isUserReady = isAudioReady(userAudioSource, userAudioData);
  const isFormValid = isReferenceReady && isUserReady;

  const handleAnalyzeClick = () => {
    if (isFormValid) {
      console.log("Analyzing...");
      // Proceed with analysis
      console.log("Reference:", referenceAudioData);
      console.log("User:", userAudioData);
    }
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

      <MultiSelectCard
        question="select target technique"
        options={VOICE_OPTIONS}
        columns={3}
        allowOther={false}
      />

      <SecondaryButton
        className={`h-fit text-xl tracking-widest transition-all duration-200 ${
          !isFormValid ? "opacity-50 cursor-not-allowed" : "hover:scale-105"
        }`}
        onClick={handleAnalyzeClick}
        disabled={!isFormValid}
      >
        analyze similarity
      </SecondaryButton>
    </div>
  );
};

export default MultiAudio;
