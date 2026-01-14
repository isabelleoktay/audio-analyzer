import { useState, useMemo } from "react";
import SurveySection from "../survey/SurveySection.jsx";
import { uploadUserStudyExitSurvey } from "../../utils/api.js";

const FinalExitSurvey = ({ surveyData, config }) => {
  const selectedTask = surveyData?.selectedTestFlow ?? "Both";
  // current section index (0 = Usefulness, 1 = Usability, etc.)
  const [sectionIndex, setSectionIndex] = useState(0);
  // saved answers per section
  const [answers, setAnswers] = useState({});
  const selectedSectionConfig = config[sectionIndex];

  // identify the vocal techniques specific question safely
  const isVocalTechniquesQuestion = (q) =>
    q?.options &&
    q.options.includes("Vibrato") &&
    q.options.includes("Breathiness");

  const filterOptionsForTask = (options) => {
    const t = (selectedTask || "").toLowerCase();
    if (t.includes("pitch") || t.includes("modulation"))
      return options.slice(0, 2);
    if (t.includes("vocal") || t.includes("tone")) return options.slice(-2);
    if (t.includes("both")) return options;
    return options; // default: show all
  };

  // compose the config to pass to SurveySection
  const composedConfig = useMemo(() => {
    const general = selectedSectionConfig?.generalQuestions || [];
    const specific = selectedSectionConfig?.specificQuestions || [];
    const processedSpecific = specific.map((q) => {
      if (isVocalTechniquesQuestion(q)) {
        return { ...q, options: filterOptionsForTask(q.options) };
      }
      return q;
    });
    return [...general, ...processedSpecific];
  }, [selectedSectionConfig, selectedTask]);

  const handleNext = (newAnswers) => {
    console.log(
      `Exit Survey answers for section ${sectionIndex + 1}:`,
      newAnswers
    );
    setAnswers((prev) => ({ ...prev, [sectionIndex]: newAnswers }));

    // If this was the last section, go home
    if (sectionIndex >= config.length - 1) {
      handleSubmitExitSurvey({ ...answers, [sectionIndex]: newAnswers });
    }

    // Otherwise, move to the next section
    setSectionIndex((prev) => prev + 1);
    window.scrollTo(0, 0);
  };

  const handleSubmitExitSurvey = async (allAnswers) => {
    try {
      const response = await uploadUserStudyExitSurvey(
        surveyData.subjectId,
        allAnswers
      );
      console.log("Exit survey uploaded successfully:", response);
    } catch (error) {
      console.error("Error uploading exit survey:", error);
    }
  };

  const handleBack = (currentAnswers) => {
    // Save the current section's answers
    setAnswers((prev) => ({
      ...prev,
      [sectionIndex]: currentAnswers,
    }));

    // Go back one section
    setSectionIndex((prev) => Math.max(prev - 1, 0));
    window.scrollTo(0, 0);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray w-full max-w-5xl px-4">
      <h1 className="text-xl text-lightpink font-bold mb-5 pt-10 text-justify">
        {selectedSectionConfig?.section}
      </h1>

      <div className="md:w-full">
        <p className="pb-5 text-justify">{selectedSectionConfig?.infoText}</p>

        <SurveySection
          config={composedConfig}
          onSubmit={handleNext}
          buttonText={sectionIndex < config.length - 1 ? "Next" : "Submit"}
          backButtonClick={sectionIndex > 0 ? handleBack : undefined}
          savedAnswers={answers[sectionIndex] || {}}
        />
      </div>
    </div>
  );
};

export default FinalExitSurvey;
