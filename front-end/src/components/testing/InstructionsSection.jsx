import FeatureInstructionsList from "./FeatureInstructionsList";
import SecondaryButton from "../buttons/SecondaryButton";

const InstructionsSection = ({ testGroup, feedbackStage, handleNextStep }) => {
  console.log("testGroup", testGroup);
  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8">
        {testGroup === "feedback" ? `Round A` : "Round B"}
      </h1>
      <div className="flex flex-col items-start justify-center w-full md:w-1/2 text-justify space-y-6 mb-8">
        {testGroup === "feedback" ? (
          feedbackStage === "before" ? (
            <>
              <p>
                In this round, you will be provided with multiple reference
                audios of short musical phrases. Each phrase corresponds with
                one of the following audio features:
              </p>
              <ul className="flex flex-col font-bold self-start space-y-2 ml-20">
                <li>Pitch</li>
                <li>Dynamics (Loudness)</li>
                <li>Tempo (Timing)</li>
              </ul>
              <p>
                For each feature, you will be asked to complete the following
                tasks:
              </p>
              <ol className="flex flex-col self-start space-y-2 ml-20">
                <li>
                  <span className="text-lightpink font-bold text-lg">
                    Record.
                  </span>{" "}
                  Listen carefully to the reference audio. Record yourself
                  replicating the reference audio as close as possible. You will
                  be given up to three attempts to reproduce the musical phrase
                  of the reference audio, but you do not have to use up all your
                  attempts.
                </li>
                <li>
                  <span className="text-lightpink font-bold text-lg">
                    Visualize.{" "}
                  </span>{" "}
                  Interact with an audio feature visualization tool. You are
                  asked to record yourself reproducing the reference audio and
                  use the available audio analysis tool to reflect on your
                  recordings. You are given a maximum 5 minute window to use the
                  audio analysis tool. Instructions for how to use the tool will
                  be provided when you reach this step.
                </li>
                <li>
                  <span className="text-lightpink font-bold text-lg">
                    Record.{" "}
                  </span>{" "}
                  Once more, record yourself replicating the same reference
                  audio as close as possible. You will be given up to three
                  attempts to reproduce the musical phrase of the reference
                  audio, but you do not have to use up all your attempts.
                </li>
              </ol>

              <p className="font-bold">
                If you are ready, please proceed to this task by clicking next.
              </p>
            </>
          ) : feedbackStage === "during" ? (
            <>
              <p>
                In this round, you will be provided with a reference audio
                example of a short musical phrase. You will first be asked to
                record yourself following the reference audio based on the
                following three parameters:
              </p>
              <ul className="flex flex-col font-bold self-start space-y-2 ml-20">
                <li>Pitch</li>
                <li>Dynamics (Loudness)</li>
                <li>Tempo (Timing)</li>
              </ul>
              <p>
                You will now have access to a visualization tool that displays
                pitch, dynamics, and tempo features extracted from your audio.
                <span className="font-bold">
                  {" "}
                  You will have up to 10 minutes to use this tool to experiment,
                  practice, and review more recordings using the same reference
                  audio.
                </span>{" "}
                You may record yourself as many times as you like during this
                practice period.
              </p>
              <p>
                You may stop using the tool whenever you feel ready to proceed
                to the next step, but you must use the tool for at least 2
                minutes before proceeding. If you are ready to proceed to the
                next task before the 10 minutes are up, you can click{" "}
                <span className="font-bold">proceed to next task</span> on the
                visualization page.{" "}
                <span className="font-bold">
                  If you are ready to proceed to the visualization tool, please
                  click next.
                </span>
              </p>
            </>
          ) : (
            <>
              <p>
                In this round, you will be provided with the same reference
                audio. Your task is to listen carefully to the example and
                attempt to reproduce it vocally. You will go through{" "}
                <span className="font-bold">three separate tasks</span>, each
                with up to{" "}
                <span className="font-bold">three recording attempts</span>{" "}
                based on the following three parameters:
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
          )
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
