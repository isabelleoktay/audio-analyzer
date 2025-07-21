import AudioPlayback from "./AudioPlayback";

const TestingSection = ({
  completedGroupsRef,
  currentTestFeature,
  attemptCount,
  testGroup,
  feedbackStage,
  children,
  audioUrl,
}) => {
  const selectInstruction = () => {
    if (currentTestFeature === "pitch") {
      return "sing the musical phrase as closely as possible to the pitch of the reference audio";
    } else if (currentTestFeature === "dynamics") {
      return "sing the musical phrase following the loudness levels of the reference audio";
    } else if (currentTestFeature === "tempo") {
      return "sing the musical phrase following the tempo changes of the reference audio";
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full md:w-1/2 space-y-6">
      <div className="flex flex-col self-start mb-2 md:mb-8 space-y-2">
        <div className="flex flex-col items-start w-full space-y-0">
          <div className="text-2xl md:text-4xl text-electricblue font-bold">
            {currentTestFeature === "pitch"
              ? "Pitch"
              : currentTestFeature === "dynamics"
              ? "Dynamics"
              : "Tempo"}
            {` - ${feedbackStage === "before" ? "Before" : "After"} Practice`}
          </div>
          <div className="text-lg md:text-xl text-lightgray font-bold">
            {testGroup === "none" ? "Without" : "With"} Feedback Tool
          </div>
        </div>
        <div className="text-sm md:text-base text-justify">
          {feedbackStage === "after" &&
            `Given your experience in the previous round, you will now be asked to record yourself once more with respect to ${currentTestFeature}. `}
          You will have three attempts to reproduce the reference audio.{" "}
          <span className="font-bold text-lightpink">
            Your goal is to {selectInstruction()}
          </span>
          . You may sing each note in the reference audio on a consonant sound
          (la, na, etc.). Once you submit your recording, you will be
          automatically directed to a short questionnaire before moving on to
          the{" "}
          {feedbackStage === "before"
            ? `practice round ${
                testGroup === "feedback" ? "with" : "without"
              } the feedback tool`
            : currentTestFeature === "tempo"
            ? completedGroupsRef.current.length === 0
              ? "instructions for the next test stage"
              : "final questionnaire"
            : "next audio feature"}
          .
        </div>
      </div>
      <div className="flex flex-col items-start w-full space-y-2">
        <div className="flex flex-row items-end justify-between w-full">
          <div className="text-lg md:text-xl font-semibold">
            Reference Audio
          </div>
          <div className="text-sm text-warmyellow bg-bluegray rounded-2xl px-2 py-1 md:px-4 md:py-2 w-fit">
            <span className="font-bold">attempts remaining:</span>{" "}
            {3 - attemptCount}
          </div>
        </div>
        <div className="w-full bg-lightgray/25 p-4 rounded-3xl">
          <AudioPlayback audioUrl={audioUrl} />
        </div>
      </div>
      {children}
    </div>
  );
};

export default TestingSection;
