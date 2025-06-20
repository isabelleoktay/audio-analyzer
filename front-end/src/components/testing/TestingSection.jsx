import { useCallback } from "react";
import ResponsiveWaveformPlayer from "../visualizations/ResponsiveWaveformPlayer";

const TestingSection = ({
  currentTestFeature,
  attemptCount,
  testGroup,
  children,
}) => {
  const getAudioUrl = useCallback(() => {
    // Assuming your audio files are in the public folder
    // You can adjust the path structure based on your actual file organization
    if (testGroup === "feedback") {
      return `/audio/twinkle_twinkle_little_star_g.m4a`;
    } else if (testGroup === "none") {
      return `/audio/mary_had_a_little_lamb_g.m4a`;
    }
    // Default fallback
    return `/audio/twinkle_twinkle_little_star_g.m4a`;
  }, [testGroup]);

  const audioUrl = getAudioUrl();

  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray w-1/2 space-y-6">
      <div className="flex flex-col self-start mb-8 space-y-2">
        <div className="text-4xl text-electricblue font-bold">
          {currentTestFeature === "pitch"
            ? "Pitch Accuracy"
            : currentTestFeature === "dynamics"
            ? "Constant Dynamics"
            : "Constant Tempo"}
        </div>
        <div>
          {currentTestFeature === "pitch"
            ? "Sing the phrase as closely as possible to the correct pitch shown in the reference audio."
            : currentTestFeature === "dynamics"
            ? "Sing the phrase with consistent loudness (volume) from start to finish."
            : "Sing the phrase at a steady speed, matching the tempo of the reference audio."}
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
