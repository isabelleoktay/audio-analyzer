import { useState } from "react";
import { useNavigate } from "react-router-dom";
import SurveySection from "../components/survey/SurveySection.jsx";
import { uploadFeedback } from "../utils/api.js";
import {
  feedbackForm1Config,
  feedbackForm2Config,
  feedbackForm3Config,
} from "../data/feedbackFormConfig.js";

const FeedbackForm = () => {
  const [step, setStep] = useState(0);
  const [showThankYou, setShowThankYou] = useState(false);
  const [answers, setAnswers] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const navigate = useNavigate();

  const handleNext = (newAnswers) => {
    console.log(`Feedback Form answers for step ${step + 1}:`, newAnswers);
    setAnswers((prev) => ({ ...prev, [step]: newAnswers }));

    // If this was the last step, go home
    if (step >= surveySections.length - 1) {
      handleSubmitFeedback({ ...answers, [step]: newAnswers });
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

  const handleSubmitFeedback = async (allAnswers) => {
    setIsSubmitting(true);
    setSubmitError(null);

    try {
      const response = await uploadFeedback(allAnswers);
      console.log("Feedback uploaded successfully:", response);

      setShowThankYou(true);
      // Wait 2 seconds, then reroute
      setTimeout(() => {
        navigate("/");
      }, 1500);
    } catch (error) {
      console.error("Error submitting feedback:", error);
      setSubmitError("Failed to submit feedback. Please try again.");
      setIsSubmitting(false);
    }
  };

  const handleBack = (currentAnswers) => {
    // Save the current section's answers
    setAnswers((prev) => ({
      ...prev,
      [step]: currentAnswers,
    }));

    // Go back one step
    setStep((prev) => Math.max(prev - 1, 0));
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
          {submitError && (
            <div className="mb-4 px-4 py-2 bg-red-500/20 border-2 border-red-500 rounded-lg text-red-400">
              {submitError}
            </div>
          )}
          <SurveySection
            config={surveySections[step].config}
            onSubmit={handleNext}
            sectionTitle={`${surveySections[step].title} - Section ${
              step + 1
            } of ${surveySections.length}`}
            buttonText={step < surveySections.length - 1 ? "Next" : "Submit"}
            backButtonClick={handleBack}
            savedAnswers={answers[step] || {}}
            isLoading={isSubmitting}
          />
        </div>
      )}
    </div>
  );
};

export default FeedbackForm;
