import { useState, useEffect } from "react";
import InstructionCard from "../cards/InstructionCard.jsx";
import UploadAudioCard from "../cards/UploadAudioCard.jsx";
import MultiSelectCard from "../cards/MultiSelectCard.jsx";
import SecondaryButton from "../buttons/SecondaryButton.jsx";

const MusaUploadAudioSection = ({ onProceed }) => {
  const [referenceAudioSource, setReferenceAudioSource] = useState(null);
  const [referenceAudioData, setReferenceAudioData] = useState(null);
  const [userAudioSource, setUserAudioSource] = useState(null);
  const [userAudioData, setUserAudioData] = useState(null);
  const [selectedVoiceType, setSelectedVoiceType] = useState([]);
  const [selectedTechniques, setSelectedTechniques] = useState([]);
  const [isFormValid, setIsFormValid] = useState(false);

  // Helper to check if audio is ready
  const isAudioReady = (audioSource, audioData) => {
    if (audioSource === "upload") {
      return audioData?.file != null;
    }
    if (audioSource === "record") {
      return audioData?.blob != null;
    }
    return false;
  };

  // Sync form validity
  useEffect(() => {
    const isReferenceReady = isAudioReady(
      referenceAudioSource,
      referenceAudioData
    );
    const isUserReady = isAudioReady(userAudioSource, userAudioData);
    const isVoiceTypeSelected =
      selectedVoiceType && selectedVoiceType.length > 0;
    const isTechniquesSelected =
      selectedTechniques && selectedTechniques.length > 0;

    setIsFormValid(
      isReferenceReady &&
        isUserReady &&
        isVoiceTypeSelected &&
        isTechniquesSelected
    );
  }, [
    referenceAudioSource,
    referenceAudioData,
    userAudioSource,
    userAudioData,
    selectedVoiceType,
    selectedTechniques,
  ]);

  // Handlers for audio source and data
  const handleAudioSourceChange = (source, type) => {
    if (type === "reference") {
      setReferenceAudioSource(source);
    } else if (type === "user") {
      setUserAudioSource(source);
    }
  };

  const handleAudioDataChange = (audioData, type) => {
    if (type === "reference") {
      setReferenceAudioData(audioData);
      if (audioData?.source === "upload" && !referenceAudioSource) {
        setReferenceAudioSource("upload");
      }
    } else if (type === "user") {
      setUserAudioData(audioData);
      if (audioData?.source === "upload" && !userAudioSource) {
        setUserAudioSource("upload");
      }
    }
  };

  const handleProceed = () => {
    if (isFormValid) {
      onProceed?.({
        userAudioData,
        referenceAudioData,
        selectedVoiceType,
        selectedTechniques,
        userAudioSource,
        referenceAudioSource,
      });
    }
  };

  return (
    <div className="flex flex-col gap-8 items-center">
      <div className="flex flex-row w-full gap-8 justify-center items-stretch min-h-[400px]">
        <div className="flex flex-col w-full gap-3">
          <InstructionCard
            title="upload your input and reference audios"
            description="the system will analyse how closely your input audio matches the vocal techniques in the uploaded/recorded reference audio."
          />
          <UploadAudioCard
            label="reference audio"
            onAudioSourceChange={(source) =>
              handleAudioSourceChange(source, "reference")
            }
            onAudioDataChange={(data) =>
              handleAudioDataChange(data, "reference")
            }
          />
          <UploadAudioCard
            label="input audio"
            onAudioSourceChange={(source) =>
              handleAudioSourceChange(source, "user")
            }
            onAudioDataChange={(data) => handleAudioDataChange(data, "user")}
          />
        </div>
        <div className="self-stretch w-px bg-darkgray/70"></div>
        <div className="flex flex-col gap-3">
          <InstructionCard
            title="select vocal specifics"
            description="the system will use this information to calibrate the analysis."
          />
          <div className="flex flex-row w-full gap-2">
            <MultiSelectCard
              question="select voice type of input audio:"
              options={["bass", "tenor", "alto", "soprano"]}
              allowOther={false}
              background_color="bg-white/10"
              onChange={setSelectedVoiceType}
            />
            <MultiSelectCard
              question="select target vocal techniques:"
              options={[
                "vibrato",
                "straight",
                "trill",
                "trillo",
                "breathy tone",
                "belting tone",
                "spoken tone",
                "inhaled singing",
                "vocal fry",
              ]}
              allowOther={false}
              background_color="bg-white/10"
              onChange={setSelectedTechniques}
            />
          </div>
        </div>
      </div>
      <SecondaryButton
        className={`h-fit w-fit text-xl transition-all duration-200`}
        isActive={isFormValid}
        onClick={handleProceed}
      >
        proceed to audio analysis
      </SecondaryButton>
    </div>
  );
};

export default MusaUploadAudioSection;
