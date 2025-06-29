import ResponsiveWaveformPlayer from "../visualizations/ResponsiveWaveformPlayer";

const TestingSection = ({
  completedGroupsRef,
  currentTestFeature,
  attemptCount,
  testGroup,
  feedbackStage,
  children,
  audioUrl,
}) => {
  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray w-1/2 space-y-6">
      <div className="flex flex-col self-start mb-8 space-y-2">
        <div className="text-4xl text-electricblue font-bold">
          {currentTestFeature === "pitch"
            ? "Pitch"
            : currentTestFeature === "dynamics"
            ? "Dynamics"
            : "Tempo"}
          {testGroup === "feedback" &&
            `${
              feedbackStage === "before"
                ? " - Before Visualizer"
                : " - After Visualizer"
            }`}
        </div>
        <div className="text-justify">
          {feedbackStage === "after" &&
            `Given your experience in the previous round, you will now be asked to record yourself once more with respect to ${currentTestFeature}. `}
          You will have three attempts to reproduce the reference audio. You
          make sing each note in the reference audio on a consonant sound (la,
          na, etc.). Once you submit your recording, you will be automatically
          directed to a short questionnaire before moving on to the{" "}
          {feedbackStage === "before" && testGroup === "feedback"
            ? "visualization tool"
            : currentTestFeature === "tempo"
            ? completedGroupsRef.current.length === 0
              ? "instructions for the next testing round"
              : "final questionnaire"
            : "next reference audio"}
          .
        </div>
      </div>
      <div className="flex flex-col items-start w-full space-y-2">
        <div className="flex flex-row items-end justify-between w-full">
          <div className="text-xl font-semibold">Reference Audio</div>
          <div className="text-sm text-warmyellow bg-bluegray rounded-2xl px-4 py-2 w-fit">
            <span className="font-bold">attempts remaining:</span>{" "}
            {3 - attemptCount}
          </div>
        </div>
        <div className="h-[100px] w-full bg-lightgray/25 p-4 rounded-3xl">
          <ResponsiveWaveformPlayer
            highlightedSections={[]}
            audioUrl={audioUrl}
          />
        </div>
      </div>
      {children}
    </div>
  );
};

export default TestingSection;
