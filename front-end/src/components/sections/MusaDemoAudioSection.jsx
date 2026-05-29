import { useState, useEffect } from "react";
import DemoAudioCard from "../cards/DemoAudioCard.jsx";
import SecondaryButton from "../buttons/SecondaryButton.jsx";
import AudioSelector from "../selectors/AudioSelector";
import SurveySingleSelect from "../survey/SurveySingleSelect.jsx";

const VOICE_CATEGORY_MAP = {
  soprano: "female",
  mezzo: "female",
  alto: "female",
  tenor: "male",
  bass: "male",
};

const MusaDemoAudioSection = ({ onProceed }) => {
  const [referenceAudioSource, setReferenceAudioSource] = useState(null);
  const [referenceAudioData, setReferenceAudioData] = useState(null);
  const [referenceVoiceType, setReferenceVoiceType] = useState([]);
  const [userAudioSource, setUserAudioSource] = useState(null);
  const [userAudioData, setUserAudioData] = useState(null);
  const [userVoiceType, setUserVoiceType] = useState([]);
  const [isFormValid, setIsFormValid] = useState(false);
  const [selectedDemoTask, setSelectedDemoTask] = useState("vocalTone");

  const vocalToneText = "Vocal Tone control (belt-breathy)";
  const pitchText = "Pitch Modulation control (vibrato-straight)";

  const getTaskText = (taskId) =>
    taskId === "vocalTone" ? vocalToneText : pitchText;

  // Derive voice category from reference voice type for input filtering
  const referenceVoiceCategory =
    referenceVoiceType.length > 0
      ? (VOICE_CATEGORY_MAP[referenceVoiceType[0]] ?? null)
      : null;

  const isAudioReady = (audioSource, audioData) => {
    if (audioSource === "upload") return audioData?.file != null;
    if (audioSource === "record") return audioData?.blob != null;
    if (audioSource === "presets") return audioData?.url != null;
    return false;
  };

  useEffect(() => {
    const isReferenceReady = isAudioReady(
      referenceAudioSource,
      referenceAudioData,
    );
    const isUserReady = isAudioReady(userAudioSource, userAudioData);

    // Voice type always required for reference (preset provides it, recording user picks it)
    const isReferenceVoiceTypeValid =
      referenceVoiceType && referenceVoiceType.length > 0;

    // For input recording: voice type is locked to reference so it's always valid once reference is set
    // For input presets: voice type comes from preset automatically
    const isUserVoiceTypeValid = userVoiceType && userVoiceType.length > 0;

    setIsFormValid(
      isReferenceReady &&
        isUserReady &&
        isReferenceVoiceTypeValid &&
        isUserVoiceTypeValid,
    );
  }, [
    referenceAudioSource,
    referenceAudioData,
    referenceVoiceType,
    userAudioSource,
    userAudioData,
    userVoiceType,
  ]);

  const handleAudioSourceChange = (source, type) => {
    if (type === "reference") {
      setReferenceAudioSource(source);
      if (source === "presets" || source === "record") {
        setReferenceVoiceType([]);
      }
    } else if (type === "user") {
      setUserAudioSource(source);
      if (source === "presets" || source === "record") {
        setUserVoiceType([]);
      }
    }
  };

  const handleAudioDataChange = (audioData, type) => {
    if (type === "reference") {
      setReferenceAudioData(audioData);
      if (audioData?.source === "presets" && !referenceAudioSource) {
        setReferenceAudioSource("presets");
      }
      if (audioData?.voiceType) {
        // Preset: auto-set voice type from metadata
        setReferenceVoiceType([audioData.voiceType]);
      } else if (audioData?.source === "record") {
        setReferenceVoiceType([]);
      }
    } else if (type === "user") {
      setUserAudioData(audioData);
      if (audioData?.source === "presets" && !userAudioSource) {
        setUserAudioSource("presets");
      }
      if (audioData?.voiceType) {
        // Preset: auto-set voice type from metadata
        setUserVoiceType([audioData.voiceType]);
      } else if (audioData?.source === "record") {
        // Recording: lock to reference voice type automatically
        if (referenceVoiceType.length > 0) {
          setUserVoiceType([...referenceVoiceType]);
        }
      }
    }
  };

  // When reference voice type changes and user is recording, sync user voice type to match
  useEffect(() => {
    if (userAudioSource === "record" && referenceVoiceType.length > 0) {
      setUserVoiceType([...referenceVoiceType]);
    }
  }, [referenceVoiceType, userAudioSource]);

  const handleProceed = () => {
    if (isFormValid) {
      onProceed?.({
        userAudioData,
        referenceAudioData,
        referenceVoiceType:
          referenceVoiceType.length > 0 ? referenceVoiceType : null,
        userVoiceType: userVoiceType.length > 0 ? userVoiceType : null,
        userAudioSource,
        referenceAudioSource,
        selectedDemoTask: getTaskText(selectedDemoTask),
      });
    }
  };

  return (
    <div className="flex flex-col w-2/3 gap-10 items-center pt-20">
      <div className="flex flex-row w-full gap-20 justify-center items-stretch min-h-[400px]">
        <div className="flex flex-col w-full gap-3">
          {/* Reference card — no filter, any voice type allowed */}
          <DemoAudioCard
            label="step 1. select or record reference audio"
            onAudioSourceChange={(source) =>
              handleAudioSourceChange(source, "reference")
            }
            onAudioDataChange={(data) =>
              handleAudioDataChange(data, "reference")
            }
            filterVoiceCategory={null}
          />

          {/* Voice type selector for reference recording */}
          {referenceAudioSource === "record" && (
            <div className="w-full items-stretch gap-2">
              <h3 className="text-sm text-lightpink">
                Reference audio voice type
              </h3>
              <SurveySingleSelect
                options={["soprano", "mezzo", "alto", "tenor", "bass"]}
                allowOther={false}
                background_color="bg-white/10"
                onChange={setReferenceVoiceType}
              />
            </div>
          )}

          {/* Input card — filtered to same category as reference */}
          <DemoAudioCard
            label="step 2. select or record input audio"
            onAudioSourceChange={(source) =>
              handleAudioSourceChange(source, "user")
            }
            onAudioDataChange={(data) => handleAudioDataChange(data, "user")}
            filterVoiceCategory={referenceVoiceCategory}
          />

          {/* Input recording: show locked voice type info, no selector */}
          {userAudioSource === "record" && referenceVoiceType.length > 0 && (
            <div className="w-full items-stretch gap-2">
              <h3 className="text-sm text-lightpink">Input audio voice type</h3>
              <div className="px-4 py-2 bg-white/10 rounded-xl text-lightgray text-sm">
                Locked to{" "}
                <span className="text-lightpink font-medium">
                  {referenceVoiceType[0]}
                </span>{" "}
                to match reference
              </div>
            </div>
          )}

          {/* Input recording with no reference yet: prompt user to select reference first */}
          {userAudioSource === "record" && referenceVoiceType.length === 0 && (
            <div className="w-full">
              <div className="px-4 py-2 bg-white/5 rounded-xl text-lightgray/50 text-sm italic">
                Select a reference audio first to set the voice type
              </div>
            </div>
          )}
        </div>
      </div>

      <SecondaryButton
        className={`h-fit w-fit text-xl transition-all duration-200`}
        isActive={isFormValid}
        onClick={handleProceed}
      >
        step 3. proceed to audio analysis
      </SecondaryButton>
    </div>
  );
};

export default MusaDemoAudioSection;
