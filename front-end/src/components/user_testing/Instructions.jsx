import SurveySingleSelect from "../survey/SurveySingleSelect.jsx";
import SecondaryButton from "../buttons/SecondaryButton.jsx";
import { useState } from "react";

const Instructions = ({ onNext, surveyData, config, configIndex }) => {
  const [confidence, setConfidence] = useState(null);

  // Use configIndex if provided, otherwise fall back to looking up by selectedTestFlow
  const currentTaskConfig =
    configIndex !== undefined
      ? config?.[configIndex]
      : config?.find((task) => task.task === surveyData?.selectedTestFlow) ??
        config?.[0] ?? {
          task: "",
          textBlock: "",
          question: "",
        };

  const startTestingProcedure = () => {
    onNext({
      instructionConfidence: confidence,
    });
    window.scrollTo(0, 0);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full">
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
