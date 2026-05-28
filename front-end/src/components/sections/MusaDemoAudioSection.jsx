import { useState, useEffect } from "react";
import DemoAudioCard from "../cards/DemoAudioCard.jsx";
import SecondaryButton from "../buttons/SecondaryButton.jsx";
import AudioSelector from "../selectors/AudioSelector";
import SurveySingleSelect from "../survey/SurveySingleSelect.jsx";

const MusaDemoAudioSection = ({ onProceed }) => {
  const [referenceAudioSource, setReferenceAudioSource] = useState(null);
  const [referenceAudioData, setReferenceAudioData] = useState(null);
  const [referenceVoiceType, setReferenceVoiceType] = useState([]);
  const [userAudioSource, setUserAudioSource] = useState(null);
  const [userAudioData, setUserAudioData] = useState(null);
  const [userVoiceType, setUserVoiceType] = useState([]);
  const [isFormValid, setIsFormValid] = useState(false);
  const [selectedDemoTask, setSelectedDemoTask] = useState("vocalTone");

  const vocalToneText = "Vocal Tone control (belt-breathy)"
  const pitchText = "Pitch Modulation control (vibrato-straight)"

  // Helper to get full task text from identifier
  const getTaskText = (taskId) => {
    return taskId === "vocalTone" ? vocalToneText : pitchText;
  };

  // Helper to check if audio is ready
  const isAudioReady = (audioSource, audioData) => {
    if (audioSource === "upload") {
      return audioData?.file != null;
    }
    if (audioSource === "record") {
      return audioData?.blob != null;
    }
    if (audioSource === "presets") {
      return audioData?.url != null;
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
    
    // Voice type only required when audio is from recording
    const isReferenceVoiceTypeRequired = referenceAudioSource === "record";
    const isReferenceVoiceTypeValid = !isReferenceVoiceTypeRequired || (referenceVoiceType && referenceVoiceType.length > 0);
    
    const isUserVoiceTypeRequired = userAudioSource === "record";
    const isUserVoiceTypeValid = !isUserVoiceTypeRequired || (userVoiceType && userVoiceType.length > 0);

    setIsFormValid(
      isReferenceReady &&
        isUserReady &&
        isReferenceVoiceTypeValid &&
        isUserVoiceTypeValid
    );
  }, [
    referenceAudioSource,
    referenceAudioData,
    referenceVoiceType,
    userAudioSource,
    userAudioData,
    userVoiceType,
  ]);

  // Handlers for audio source and data
  const handleAudioSourceChange = (source, type) => {
    if (type === "reference") {
      setReferenceAudioSource(source);
      // Clear voice type when switching to presets (it's known from presets)
      if (source === "presets") {
        setReferenceVoiceType([]);
      }
    } else if (type === "user") {
      setUserAudioSource(source);
      // Clear voice type when switching to presets (it's known from presets)
      if (source === "presets") {
        setUserVoiceType([]);
      }
    }
  };

  const handleAudioDataChange = (audioData, type) => {
    if (type === "reference") {
      setReferenceAudioData(audioData);
      if (audioData?.source === "presets" && !referenceAudioSource) {
        setReferenceAudioSource("presets");
        setReferenceVoiceType([]);
      }
    } else if (type === "user") {
      setUserAudioData(audioData);
      if (audioData?.source === "presets" && !userAudioSource) {
        setUserAudioSource("presets");
        setUserVoiceType([]);
      }
    }
  };

  const handleSelectDemoTask = (taskOption) => {
    setSelectedDemoTask(taskOption);
  };

  const handleProceed = () => {
    if (isFormValid) {
      onProceed?.({
        userAudioData,
        referenceAudioData,
        referenceVoiceType: referenceAudioSource === "record" ? referenceVoiceType : null,
        userVoiceType: userAudioSource === "record" ? userVoiceType : null,
        userAudioSource,
        referenceAudioSource,
        selectedDemoTask: getTaskText(selectedDemoTask),
      });
    }
  };

  return (
    <div className="flex flex-col w-2/3 gap-10 items-center pt-20">
      {/* <div className="w-full items-stretch gap-10 h-[50px]">
        <h1 className="text-2xl text-lightpink">
          Select task to analyse
        </h1>
        <AudioSelector
              selectedOption={selectedDemoTask}
              onSourceChange={handleSelectDemoTask}
              option1 = "vocalTone"
              option2 = "pitch"
              option1Text = {vocalToneText}
              option2Text = {pitchText}
            />
      </div> */}

      <div className="flex flex-row w-full gap-20 justify-center items-stretch min-h-[400px]">
        <div className="flex flex-col w-full gap-3">
          <DemoAudioCard
            label="reference audio"
            onAudioSourceChange={(source) =>
              handleAudioSourceChange(source, "reference")
            }
            onAudioDataChange={(data) =>
              handleAudioDataChange(data, "reference")
            }
          />
          
          {/* Voice type selector for reference audio when recording */}
          {referenceAudioSource === "record" && (
            <div className="w-full items-stretch gap-2">
              <h3 className="text-sm text-lightpink">
                Reference audio voice type
              </h3>
              <SurveySingleSelect
                  options={["bass", "tenor", "alto", "soprano"]}
                  allowOther={false}
                  background_color="bg-white/10"
                  onChange={setReferenceVoiceType}
                />
            </div>
          )}

          <DemoAudioCard
            label="input audio"
            onAudioSourceChange={(source) =>
              handleAudioSourceChange(source, "user")
            }
            onAudioDataChange={(data) => handleAudioDataChange(data, "user")}
          />

          {/* Voice type selector for user audio when recording */}
          {userAudioSource === "record" && (
            <div className="w-full items-stretch gap-2">
              <h3 className="text-sm text-lightpink">
                Input audio voice type
              </h3>
              <SurveySingleSelect
                  options={["bass", "tenor", "alto", "soprano"]}
                  allowOther={false}
                  background_color="bg-white/10"
                  onChange={setUserVoiceType}
                />
            </div>
          )}
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

export default MusaDemoAudioSection;