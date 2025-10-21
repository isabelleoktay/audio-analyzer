import { useState } from "react";
import { useNavigate } from "react-router-dom";
import SurveySection from "../components/survey/SurveySection.jsx";
import {
  feedbackForm1Config,
  feedbackForm2Config,
  feedbackForm3Config,
} from "../data/feedbackFormConfig.js";

const FeedbackForm = () => {
  const [step, setStep] = useState(0);
  const navigate = useNavigate();
  const [showThankYou, setShowThankYou] = useState(false);

  const handleNext = (answers) => {
    console.log(`Feedback Form answers for step ${step + 1}:`, answers);

    // If this was the last step, go home
    if (step >= surveySections.length - 1) {
      setShowThankYou(true);
      // Wait 2 seconds, then reroute
      setTimeout(() => {
        navigate("/");
      }, 1500);
    }

    // Otherwise, move to the next section
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

  return (
    <div className="flex items-center justify-center min-h-screen">
      {showThankYou ? (
        <h1 className="text-5xl font-bold text-lightpink animate-zoomIn text-center">
          Thank you for providing feedback to the MuSA Development team!
        </h1>
      ) : (
        <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
          <SurveySection
            config={surveySections[step].config}
            onSubmit={handleNext}
            sectionTitle={`${surveySections[step].title} - Section ${
              step + 1
            } of ${surveySections.length}`}
            buttonText={step < surveySections.length - 1 ? "Next" : "Submit"}
            backButtonClick={
              step > 0 ? () => setStep((prev) => prev - 1) : null
            }
          />
        </div>
      )}
    </div>
  );
};

export default FeedbackForm;
