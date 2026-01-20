import { useState } from "react";

import SecondaryButton from "../../components/buttons/SecondaryButton.jsx";
import HighlightedText from "../text/HighlightedText.jsx";
import AudioPlayButton from "./AudioPlayButton.jsx";
import AudioRecorder from "./AudioRecorder.jsx";

const RecordTask = ({
  onNext,
  config,
  configIndex,
  surveyData,
  metadata = {},
}) => {
  const [hasRecordings, setHasRecordings] = useState(false);

  const { phase = "pre-practice", condition = "control" } = metadata;
  const isPost = phase === "post-practice";

  const baseConfig = config?.[configIndex] ?? {};
  const conditionConfig = baseConfig.conditions?.[condition] ?? {};

  // Select instruction based on phase
  const displayInstruction = isPost
    ? baseConfig.instructions?.post
    : baseConfig.instructions?.pre;

  const currentTaskConfig = {
    title: baseConfig.title,
    instruction: displayInstruction,
    ...conditionConfig,
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full px-8">
      <div className="w-9/12">
        <h2 className="text-4xl text-electricblue font-bold mb-4 text-center">
          {currentTaskConfig.title} Task
        </h2>
        <p className="text-left text-lg">{currentTaskConfig.instruction}</p>
        <hr className="border-t border-lightgray/20 mb-8 mt-2" />

        <div className="flex flex-col space-y-1">
          <p className="text-left text-lg text-warmyellow font-semibold">
            Listen and sing the phrase:
          </p>
          <div className="flex flex-row items-center gap-4 bg-blueblack/50 p-3 rounded-3xl w-full">
            <AudioPlayButton audioUrl="https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3" />
            <HighlightedText
              text={currentTaskConfig.phrase}
              highlightWords={currentTaskConfig.highlightedText}
              highlightClass={currentTaskConfig.highlightClass}
              defaultClass={currentTaskConfig.defaultClass}
              highlightLabel={baseConfig.highlightLabel}
              defaultLabel={baseConfig.defaultLabel}
              highlightLabelColor={baseConfig.highlightLabelColor}
              defaultLabelColor={baseConfig.defaultLabelColor}
              className="text-center justify-center flex-grow"
            />
          </div>
        </div>
        <div className="flex flex-col mt-8 space-y-3">
          {/* <div>
            <p className="text-left text-lg font-semibold text-lightpink mb-1">
              Reference Audio:
            </p>
            <div className="bg-lightgray/20 p-4 rounded-3xl w-full">
              <ResponsiveWaveformPlayer
                audioUrl="https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3"
                highlightedSections={[]}
                progressColor="#FFD6E8"
                waveColor="#E0E0E0"
                showTimeline={false}
                playButtonColor="text-lightgray"
              />
            </div>
          </div> */}
          <div>
            <p className="text-left text-lg font-semibold text-lightpink mb-1">
              Your Recording:
            </p>
            <AudioRecorder
              maxAttempts={3}
              onAttemptsChange={(count) => setHasRecordings(count > 0)}
            />
          </div>
        </div>
        <hr className="border-t border-lightgray/20 mb-4 mt-8" />

        <div className="flex justify-center">
          <SecondaryButton onClick={() => onNext()} disabled={!hasRecordings}>
            {hasRecordings
              ? "Continue with your selected attempt."
              : "Please record at least one attempt to continue"}
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
};

export default RecordTask;
