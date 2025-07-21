import FeatureInstructionsList from "./FeatureInstructionsList";
import SecondaryButton from "../buttons/SecondaryButton";

const InstructionsSection = ({ testGroup, feedbackStage, handleNextStep }) => {
  console.log("testGroup", testGroup);
  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8">
        {testGroup === "feedback"
          ? `With Feedback Tool`
          : "Without Feedback Tool"}
      </h1>
      <div className="flex flex-col items-start text-sm md:text-base justify-center w-full md:w-2/3 text-justify space-y-6 mb-8">
        {
          feedbackStage === "before" ? (
            <>
              <p className="font-bold text-base md:text-lg text-left self-center">
                You will be provided with reference audios of melodic phrases.
                Each audio corresponds to one of these musical features:
              </p>
              <FeatureInstructionsList />
              <p className="font-bold text-base md:text-lg text-center self-center">
                For each feature, you will complete the following tasks:
              </p>
              <ol className="flex flex-col self-start space-y-2 ml-0">
                <li>
                  <span className="text-lightpink font-bold text-base md:text-lg">
                    Record.
                  </span>{" "}
                  Listen carefully to the reference audio. Record yourself
                  replicating the reference audio as close as possible.
                </li>
                <li>
                  <span className="text-lightpink font-bold text-base md:text-lg">
                    {testGroup === "feedback" ? `Visualize` : `Practice`}.
                  </span>{" "}
                  {testGroup === "feedback"
                    ? `Use the audio analysis feedback tool to review your recording and practice`
                    : `Review your recording and practice by yourself`}
                  .
                </li>
                <li>
                  <span className="text-lightpink font-bold text-base md:text-lg">
                    Record.
                  </span>{" "}
                  Record yourself replicating the reference audio as close as
                  possible{" "}
                  {testGroup === "feedback"
                    ? `after using the audio analysis feedback tool`
                    : `after practicing by yourself`}
                  .
                </li>
              </ol>
              <p className="font-bold text-base md:text-lg text-center self-center">
                If you are ready, please proceed by clicking next.
              </p>
            </>
          ) : (
            <div className="flex flex-col items-start gap-2">
              <p>
                {testGroup === "feedback" ? (
                  <>
                    You will now have access to a visualization tool that
                    displays pitch, dynamics, and tempo features extracted from
                    your audio.
                    <span className="font-bold">
                      {" "}
                      You will have up to 10 minutes to use this tool to
                      experiment, practice, and review more recordings using the
                      same reference audio.
                    </span>{" "}
                    You may record yourself as many times as you like during
                    this practice period. You may stop whenever you feel ready
                    to proceed to the next step, but you must use the feedback
                    tool on at least one recording.
                  </>
                ) : (
                  <p className="font-bold text-lightpink">
                    You will now proceed without access to the visualization
                    tool. Instead, you will focus on recording and reviewing
                    your attempts based on the reference audio.
                  </p>
                )}
              </p>
              <p>If you are ready to proceed, please click next.</p>
            </div>
          )

          // : (
          //   <>
          //     <p>
          //       In this round, you will be provided with the same reference audio.
          //       Your task is to listen carefully to the example and attempt to
          //       reproduce it vocally. You will go through{" "}
          //       <span className="font-bold">three separate tasks</span>, each with
          //       up to <span className="font-bold">three recording attempts</span>{" "}
          //       based on the following three parameters:
          //     </p>
          //     <FeatureInstructionsList />
          //     <p>
          //       You may listen to the reference audio as many times as you need
          //       before each attempt. After each task, you will be guided to the
          //       next one.{" "}
          //       <span className="font-bold">
          //         If you are ready, please proceed by clicking next.
          //       </span>
          //     </p>
          //   </>
          // )
        }
      </div>
      <SecondaryButton onClick={handleNextStep}>Next</SecondaryButton>
    </div>
  );
};

export default InstructionsSection;
