import { useState, useMemo } from "react";
import SurveySection from "../survey/SurveySection.jsx";
import { uploadUserStudyExitSurvey } from "../../utils/api.js";

const FinalExitSurvey = ({ surveyData, config, onNext }) => {
  const selectedTask = surveyData?.selectedTestFlow ?? "Both";

  const [sectionIndex, setSectionIndex] = useState(0);
  const [answers, setAnswers] = useState({});

  const selectedSectionConfig = config[sectionIndex];

  const isVocalTechniquesQuestion = (q) =>
    q?.options &&
    q.options.includes("Vibrato") &&
    q.options.includes("Breathiness");

  const filterOptionsForTask = (options) => {
    const t = (selectedTask || "").toLowerCase();
    if (t.includes("pitch") || t.includes("modulation"))
      return options.slice(0, 2);
    if (t.includes("vocal") || t.includes("tone")) return options.slice(-2);
    return options;
  };

  const composedConfig = useMemo(() => {
    const general = selectedSectionConfig?.generalQuestions || [];
    const specific = selectedSectionConfig?.specificQuestions || [];

    return [
      ...general,
      ...specific.map((q) =>
        isVocalTechniquesQuestion(q)
          ? { ...q, options: filterOptionsForTask(q.options) }
          : q
      ),
    ];
  }, [selectedSectionConfig, selectedTask]);

  const handleNext = (sectionAnswers) => {
    setAnswers((prev) => ({
      ...prev,
      [sectionIndex]: sectionAnswers,
    }));

    setSectionIndex((i) => i + 1);
    window.scrollTo(0, 0);
  };

  const handleBack = (sectionAnswers) => {
    setAnswers((prev) => ({
      ...prev,
      [sectionIndex]: sectionAnswers,
    }));

    setSectionIndex((i) => Math.max(i - 1, 0));
    window.scrollTo(0, 0);
  };

  const handleFinalSubmit = async (finalSectionAnswers) => {
    const finalAnswers = {
      ...answers,
      [sectionIndex]: finalSectionAnswers,
    };

    try {
      await uploadUserStudyExitSurvey(surveyData.subjectId, finalAnswers);
      onNext?.();
    } catch (err) {
      console.error("Error uploading exit survey:", err);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen w-full max-w-5xl px-4">
      <h1 className="text-xl font-bold mb-5">
        {selectedSectionConfig?.section}
      </h1>

      <SurveySection
        key={sectionIndex}
        config={composedConfig}
        savedAnswers={answers[sectionIndex] || {}}
        onSubmit={
          sectionIndex === config.length - 1 ? handleFinalSubmit : handleNext
        }
        backButtonClick={sectionIndex > 0 ? handleBack : undefined}
        buttonText={sectionIndex === config.length - 1 ? "Submit" : "Next"}
      />
    </div>
  );
};

export default FinalExitSurvey;
