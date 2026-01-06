import SurveySingleSelect from "../../components/survey/SurveySingleSelect.jsx";
import SecondaryButton from "../../components/buttons/SecondaryButton";
import { musaVoiceTestInstructionsConfig } from "../../data/musaVoiceTestInstructionsConfig.js";
import { useState } from "react";

const Instructions = ({
  onNext,
  surveyData,
  config = musaVoiceTestInstructionsConfig,
}) => {
  const [confidence, setConfidence] = useState(null);

  const currentTask =
    surveyData?.selectedTestFlow ?? "Pitch Modulation Control";

  const currentTaskConfig = config?.find((task) => task.task === currentTask) ??
    config?.[0] ?? {
      task: "",
      textBlock: "",
      question: "",
    };

  const startTestingProcedure = () => {
    onNext({
      instructionConfidence: confidence,
    });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8 text-center">
        {currentTaskConfig.task}
      </h1>

      <div className="text-lightgrey text-justify w-full md:w-1/2">
        <p className="mb-6">{currentTaskConfig.textBlock}</p>

        <div className="mb-6 w-full aspect-video">
          <iframe
            className="w-full h-full rounded-lg"
            src={currentTaskConfig.videoUrl}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>

        <SurveySingleSelect
          question={currentTaskConfig.question}
          options={[
            "Not at all",
            "Somewhat Confident",
            "Confident",
            "Very Confident",
          ]}
          onChange={setConfidence}
        />
      </div>

      <div className="pt-10">
        <SecondaryButton onClick={startTestingProcedure} disabled={!confidence}>
          i understand what to do. continue.
        </SecondaryButton>
      </div>
    </div>
  );
};

export default Instructions;
