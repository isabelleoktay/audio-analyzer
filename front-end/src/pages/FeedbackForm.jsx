import { useState } from "react";
import { useNavigate } from "react-router-dom";
import SurveySection from "../components/survey/SurveySection.jsx";
import { uploadFeedback } from "../utils/api.js";
import {
  feedbackForm1Config,
  feedbackForm2Config,
  feedbackForm3Config,
} from "../config/feedbackFormConfig.js";

const FeedbackForm = () => {
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showThankYou, setShowThankYou] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const navigate = useNavigate();

  const surveySections = [
    {
      config: feedbackForm1Config,
      title: "Understanding Your Practice Habits",
    },
    {
      config: feedbackForm2Config,
      title: "MuSA Impact",
    },
    {
      config: feedbackForm3Config,
      title: "Technology Awareness",
    },
  ];

  const handleNext = (sectionAnswers) => {
    setAnswers((prev) => ({
      ...prev,
      [step]: sectionAnswers,
    }));

    setStep((s) => s + 1);
    window.scrollTo(0, 0);
  };

  const handleBack = (sectionAnswers) => {
    setAnswers((prev) => ({
      ...prev,
      [step]: sectionAnswers,
    }));

    setStep((s) => Math.max(s - 1, 0));
    window.scrollTo(0, 0);
  };

  const handleFinalSubmit = async (finalSectionAnswers) => {
    const finalAnswers = {
      ...answers,
      [step]: finalSectionAnswers,
    };

    setIsSubmitting(true);
    setSubmitError(null);

    try {
      await uploadFeedback(finalAnswers);
      setShowThankYou(true);

      setTimeout(() => {
        navigate("/");
      }, 1500);
    } catch (error) {
      console.error("Error submitting feedback:", error);
      setSubmitError("Failed to submit feedback. Please try again.");
      setIsSubmitting(false);
    }
  };

  if (showThankYou) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <h1 className="text-5xl font-bold text-lightpink text-center animate-zoomIn">
          Thank you for providing feedback to the MuSA Development team!
        </h1>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="w-full max-w-4xl p-8 rounded-xl pt-20">
        {submitError && (
          <div className="mb-4 px-4 py-2 bg-red-500/20 border-2 border-red-500 rounded-lg text-red-400">
            {submitError}
          </div>
        )}

        <SurveySection
          key={step} // 🔥 forces clean remount per step
          config={surveySections[step].config}
          sectionTitle={`${surveySections[step].title} – Section ${
            step + 1
          } of ${surveySections.length}`}
          savedAnswers={answers[step] || {}}
          backButtonClick={step > 0 ? handleBack : undefined}
          isLoading={isSubmitting}
          buttonText={step < surveySections.length - 1 ? "Next" : "Submit"}
          onSubmit={
            step === surveySections.length - 1 ? handleFinalSubmit : handleNext
          }
        />
      </div>
    </div>
  );
};

export default FeedbackForm;
