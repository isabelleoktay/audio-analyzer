import { useState } from "react";
import SecondaryButton from "../buttons/SecondaryButton";

const Rating = ({
  onSubmit,
  testGroup,
  currentTestFeature,
  feedbackStage,
  completedGroupsRef,
}) => {
  const [selectedValue, setSelectedValue] = useState(null);
  const [helpfulnessValue, setHelpfulnessValue] = useState(null); // New state for the second rating

  const handleChange = (event) => {
    setSelectedValue(Number(event.target.value));
  };

  const handleHelpfulnessChange = (event) => {
    setHelpfulnessValue(Number(event.target.value)); // Handle second rating
  };

  const handleSubmit = () => {
    if (selectedValue !== null) {
      if (
        feedbackStage === "after" &&
        testGroup === "feedback" &&
        currentTestFeature !== "tempo"
      ) {
        if (helpfulnessValue === null) {
          alert("Please rate the helpfulness of the highlighted sections.");
          return;
        }
        onSubmit({
          performanceRating: selectedValue,
          helpfulnessRating: helpfulnessValue,
        });
      } else {
        onSubmit({ performanceRating: selectedValue });
      }
    } else {
      alert("Please select a value before submitting.");
    }
  };

  const getRedirectMessage = () => {
    if (
      completedGroupsRef.current.length === 0 &&
      currentTestFeature === "tempo"
    ) {
      return "Once you submit your response, you will automatically be directed to the instructions for the next stage.";
    } else if (
      completedGroupsRef.current.length >= 1 &&
      currentTestFeature === "tempo"
    ) {
      return "Once you submit your response, you will be redirected to the final questionnaire.";
    } else {
      if (feedbackStage === "before") {
        return `Once you submit your response, you will automatically be directed to the practice round ${
          testGroup === "feedback" ? "with" : "without"
        } the feedback tool.`;
      }
      return "Once you submit your response, you will automatically be directed to the next audio feature.";
    }
  };

  const getRatingQuestion = () => {
    switch (currentTestFeature) {
      case "tempo":
        return "How would you rate your control of tempo (speed and steadiness) in this performance?";
      case "pitch":
        return "On a scale from 1 to 7, how well do you think you stayed in tune (sang the correct pitches) in this performance?";
      case "dynamics":
        return "How would you rate your control of dynamics (loudness and softness) in this performance?";
      default:
        return "How would you rate the audio?";
    }
  };

  const getRatingSubtitle = () => {
    switch (currentTestFeature) {
      case "tempo":
        return "(1 = Very poor / inconsistent, 7 = Very good / controlled)";
      case "pitch":
        return "(1 = Very poor / mostly out of tune, 7 = Excellent / consistently in tune)";
      case "dynamics":
        return "(1 = Very poor / flat or uncontrolled dynamics, 7 = Very good / expressive and well-controlled dynamics)";
      default:
        return "How would you rate the audio?";
    }
  };

  return (
    <div className="flex flex-col items-center justify-center text-center min-h-screen text-lightgray w-full md:w-1/2 space-y-6">
      <div className="flex flex-col items-center justify-center mb-8 space-y-8">
        <div className="flex flex-col items-start space-y-1 w-full">
          <div className="flex flex-col items-start  w-full space-y-0">
            <div className="text-4xl text-electricblue font-bold capitalize">
              {currentTestFeature} - {feedbackStage} Practice
            </div>
            <div className="text-lg md:text-xl text-lightgray font-bold">
              {testGroup === "none" ? "Without" : "With"} Feedback Tool -
              Questionnaire
            </div>
          </div>
          <div className="text-left items-start">{getRedirectMessage()}</div>
        </div>
        <div className="flex flex-col items-center justify-center w-full bg-blueblack/25 p-6 rounded-3xl">
          <div className="flex flex-col items-center justify-center w-full space-y-6">
            {/* First Rating */}
            <div className="text-lg md:text-2xl text-left text-lightgray">
              <span className="font-semibold">{getRatingQuestion()}</span>{" "}
              <span className="font-normal text-base md:text-xl italic">
                {getRatingSubtitle()}
              </span>
            </div>
            <div className="grid grid-cols-4 md:grid-cols-7 gap-4 justify-items-center">
              {Array.from({ length: 7 }, (_, index) => (
                <label
                  key={index}
                  className={`flex flex-col items-center cursor-pointer ${
                    selectedValue === index + 1
                      ? "text-lightpink font-bold"
                      : "text-gray-500"
                  }`}
                >
                  <input
                    type="radio"
                    name="rating"
                    value={index + 1}
                    checked={selectedValue === index + 1}
                    onChange={handleChange}
                    className="hidden"
                  />
                  <div
                    className={`w-7 h-7 text-sm md:text-base md:w-10 md:h-10 flex items-center justify-center rounded-full border-2 ${
                      selectedValue === index + 1
                        ? "border-lightpink bg-darkpink text-white"
                        : "border-gray-300 bg-white hover:border-darkpink hover:bg-darkpink hover:text-white"
                    }`}
                  >
                    {index + 1}
                  </div>
                </label>
              ))}
            </div>
            {/* Second Rating (Conditionally Rendered) */}
            {feedbackStage === "after" && (
              <div className="flex flex-col items-center justify-center w-full space-y-6">
                <div className="text-lg md:text-2xl text-left text-lightgray">
                  <span className="font-semibold">
                    How helpful was the practice round (
                    {testGroup === "none" ? "without" : "with"} the feedback
                    tool) in understanding your {currentTestFeature} accuracy?
                  </span>{" "}
                  <span className="font-normal text-base md:text-xl italic">
                    (1 = Not helpful at all, 7 = Extremely helpful)
                  </span>
                </div>
                <div className="grid grid-cols-4 md:grid-cols-7 gap-4 justify-items-center">
                  {Array.from({ length: 7 }, (_, index) => (
                    <label
                      key={index}
                      className={`flex flex-col items-center cursor-pointer ${
                        helpfulnessValue === index + 1
                          ? "text-lightpink font-bold"
                          : "text-gray-500"
                      }`}
                    >
                      <input
                        type="radio"
                        name="helpfulness"
                        value={index + 1}
                        checked={helpfulnessValue === index + 1}
                        onChange={handleHelpfulnessChange}
                        className="hidden"
                      />
                      <div
                        className={`w-7 h-7 text-sm md:text-base md:w-10 md:h-10 flex items-center justify-center rounded-full border-2 ${
                          helpfulnessValue === index + 1
                            ? "border-lightpink bg-darkpink text-white"
                            : "border-gray-300 bg-white hover:border-darkpink hover:bg-darkpink hover:text-white"
                        }`}
                      >
                        {index + 1}
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        <SecondaryButton className="w-fit" onClick={handleSubmit}>
          Submit
        </SecondaryButton>
      </div>
    </div>
  );
};

export default Rating;
