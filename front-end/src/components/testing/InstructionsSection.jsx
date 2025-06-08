import FeatureInstructionsList from "./FeatureInstructionsList";
import SecondaryButton from "../buttons/SecondaryButton";

const InstructionsSection = ({ testGroup, handleNextStep }) => {
  console.log("testGroup", testGroup);
  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8">
        {testGroup === "feedback" ? "Visual Feedback" : "No Visual Feedback"}
      </h1>
      <div className="flex flex-col items-start justify-center w-1/2 text-justify space-y-6 mb-8">
        {testGroup === "feedback" ? (
          <>
            <p>
              In this round, you will be provided with a reference audio example
              of a short musical phrase. During recording, you will have access
              to a visualization tool that displays features from your
              recordings, including:
            </p>
            <ul className="flex flex-col font-bold self-start space-y-2 ml-20">
              <li>Pitch</li>
              <li>Dynamics (Loudness)</li>
              <li>Tempo (Timing)</li>
            </ul>
            <p>
              You will have 10 minutes to use this tool to experiment, practice,
              and review your recordings. You may record yourself as many times
              as you like during this practice period.
            </p>
            <p>
              After this period, you will be asked to record yourself performing
              a new phrase, with the following three objectives:
            </p>
            <FeatureInstructionsList />
            <p>
              You will have up to three attempts for each of these tasks. If you
              are ready, please proceed by clicking next.
            </p>
          </>
        ) : (
          <>
            <p>
              In this round, you will be provided with a reference audio example
              of a short musical phrase. Your task is to listen carefully to the
              example and attempt to reproduce it vocally. You will go through{" "}
              <span className="font-bold">three separate tasks</span>, each with
              up to <span className="font-bold">three recording attempts</span>:
            </p>
            <FeatureInstructionsList />
            <p>
              You may listen to the reference audio as many times as you need
              before each attempt. After each task, you will be guided to the
              next one.{" "}
              <span className="font-bold">
                If you are ready, please proceed by clicking next.
              </span>
            </p>
          </>
        )}
      </div>
      <SecondaryButton onClick={handleNextStep}>Next</SecondaryButton>
    </div>
  );
};

export default InstructionsSection;
