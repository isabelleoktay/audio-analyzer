import { useState, useMemo } from "react";
import { FinalExitConfig } from "../../data/musaVoiceTestSurveysConfig";
import SurveySection from "../../components/survey/SurveySection.jsx";

const FinalExitSurvey = ({ onNext, surveyData, config = FinalExitConfig }) => {
  const selectedTask = surveyData?.selectedTestFlow ?? "Both";

  // current section index (0 = Usefulness, 1 = Usability, etc.)
  const [sectionIndex, setSectionIndex] = useState(0);

  // saved answers per section
  const [savedAnswersBySection, setSavedAnswersBySection] = useState({});

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

  const handleSubmitSection = (answers) => {
    // save current section answers
    setSavedAnswersBySection((prev) => ({ ...prev, [sectionIndex]: answers }));

    // if more sections left, move to next
    if (sectionIndex < config.length - 1) {
      setSectionIndex((i) => i + 1);
      return;
    }

    // final submit: merge all section answers
    const finalAnswers = { ...savedAnswersBySection, [sectionIndex]: answers };
    console.log("Final exit survey answers:", finalAnswers);
    onNext({
      finalExitAnswers: finalAnswers,
    });
  };

  const handleBack = (answers) => {
    setSavedAnswersBySection((prev) => ({ ...prev, [sectionIndex]: answers }));
    setSectionIndex((i) => Math.max(0, i - 1));
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
          onSubmit={handleSubmitSection}
          buttonText={sectionIndex < config.length - 1 ? "Next" : "Submit"}
          backButtonClick={sectionIndex > 0 ? handleBack : undefined}
          savedAnswers={savedAnswersBySection[sectionIndex] || {}}
        />
      </div>
    </div>
  );
};

export default FinalExitSurvey;
