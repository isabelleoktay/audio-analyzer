import SurveySingleSelect from "../survey/SurveySingleSelect.jsx";
import SecondaryButton from "../buttons/SecondaryButton.jsx";
import { useState, useEffect } from "react";
import { uploadUserStudySectionField } from "../../utils/api.js";

const Instructions = ({
  onNext,
  surveyData,
  config,
  configIndex,
  sectionKey,
}) => {
  const [confidence, setConfidence] = useState(null);

  // Reset confidence when moving to a new section (via component remount with key)
  useEffect(() => {
    setConfidence(null);
  }, [sectionKey, configIndex]);

  // Resolve config entry robustly: support numeric index or string keys (sectionKey or slug)
  const resolveConfigEntry = () => {
    if (!config) return { task: "", textBlock: "", question: "" };

    // If numeric index, use directly
    if (typeof configIndex === "number") {
      return config?.[configIndex] ?? config?.[0];
    }

    // If string, try matching against sectionKey or slugified task
    if (typeof configIndex === "string") {
      const bySection = config.find((c) => c.sectionKey === configIndex);
      if (bySection) return bySection;

      const bySectionPrefixed = config.find(
        (c) => `instructions-${c.sectionKey}` === configIndex,
      );
      if (bySectionPrefixed) return bySectionPrefixed;

      const slugMatch = config.find(
        (c) =>
          c.task && c.task.replace(/\s+/g, "-").toLowerCase() === configIndex,
      );
      if (slugMatch) return slugMatch;
    }

    // Fall back to selectedTestFlow match, then first entry
    const bySelected = config.find(
      (task) => task.task === surveyData?.selectedTestFlow,
    );
    return (
      bySelected ?? config?.[0] ?? { task: "", textBlock: "", question: "" }
    );
  };

  const currentTaskConfig = resolveConfigEntry();

  // Support new config shape where each task has a `parts` array.
  const displayPart = currentTaskConfig?.parts?.[0] ??
    currentTaskConfig ?? { textBlock: "", videoUrl: "", question: "" };

  const handleSubmit = async (confidence) => {
    try {
      await uploadUserStudySectionField({
        subjectId: surveyData.subjectId,
        sectionKey: sectionKey,
        field: "instructionConfidence",
        data: confidence,
      });

      // console.log("survey before practice answers uploaded successfully");
      onNext({
        instructionConfidence: confidence,
      });
      window.scrollTo(0, 0);
    } catch (error) {
      onNext({
        instructionConfidence: confidence,
      });
      console.error("Error uploading survey before practice answers:", error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mt-10 mb-8 text-center">
        {currentTaskConfig.task}
      </h1>

      <div className="text-lightgrey text-justify w-full md:w-1/2">
        <p className="mb-6">{displayPart.textBlock}</p>

        <div className="mb-6 w-full aspect-video">
          <iframe
            className="w-full h-full rounded-lg"
            src={displayPart.videoUrl}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>

        <SurveySingleSelect
          question={displayPart.question}
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
        <SecondaryButton
          onClick={() => handleSubmit(confidence)}
          disabled={!confidence}
        >
          i understand what to do. continue.
        </SecondaryButton>
      </div>
    </div>
  );
};

export default Instructions;
