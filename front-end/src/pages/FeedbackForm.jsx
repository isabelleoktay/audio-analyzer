import { useState, useEffect } from "react";
import Survey from "../components/survey/Survey.jsx";
import {
  feedbackForm1Config,
  feedbackForm2Config,
  feedbackForm3Config,
} from "../data/feedbackFormConfig.js";

const FeedbackForm = () => {
  const [step, setStep] = useState(0);

  const handleNext = (answers) => {
    console.log("Feedback Form answers for step", step + 1, ":", answers);
    setStep((prev) => prev + 1);
    window.scrollTo(0, 0);
  };

  const surveySections = [
    {
      config: feedbackForm1Config,
      title: "Understanding Your Practice Habits",
    },
    { config: feedbackForm2Config, title: "MuSA Impact" },
    { config: feedbackForm3Config, title: "Technology Awareness" },
  ];

  if (step >= surveySections.length) {
    // All steps completed
    return <div className="w-full h-full" />;
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
        <Survey
          config={surveySections[step].config}
          onSubmit={handleNext}
          sectionTitle={`${surveySections[step].title} - Section ${
            step + 1
          } of ${surveySections.length}`}
          buttonText={step < surveySections.length - 1 ? "Next" : "Submit"}
        />
      </div>
    </div>
  );
};

export default FeedbackForm;
