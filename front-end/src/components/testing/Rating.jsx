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

  const handleChange = (event) => {
    setSelectedValue(Number(event.target.value));
  };

  const handleSubmit = () => {
    if (selectedValue !== null) {
      onSubmit(selectedValue);
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
        return "Once you submit your response, you will automatically be directed to the visualization tool.";
      }
      return "Once you submit your response, you will automatically be directed to the next reference audio.";
    }
  };

  return (
    <div className="flex flex-col items-center justify-center text-center h-screen text-lightgray w-1/2 space-y-6">
      <div className="flex flex-col mb-8 space-y-12">
        <div className="flex flex-col w-full space-y-2">
          <div className="text-4xl text-electricblue font-bold capitalize">
            {currentTestFeature} {testGroup === "feedback" ? feedbackStage : ""}{" "}
            - Questionnaire
          </div>
          <div className="text-lightgray">
            Please answer the following. {getRedirectMessage()}
          </div>
        </div>
        <div className="flex flex-col items-center justify-center w-full space-y-6 bg-blueblack/25 p-6 rounded-3xl">
          <div className="text-2xl text-center text-lightgray">
            How would you rate your{" "}
            <span className="text-darkpink font-semibold">
              {currentTestFeature} performance
            </span>{" "}
            on a scale from 1 (lowest) to 10 (highest)?
          </div>
          <div className="flex flex-row space-x-4">
            {Array.from({ length: 10 }, (_, index) => (
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
                  className={`w-10 h-10 flex items-center justify-center rounded-full border-2 ${
                    selectedValue === index + 1
                      ? "border-lightpink bg-darkpink text-white"
                      : "border-gray-300 bg-white hover:border-lightpink hover:bg-lightpink hover:text-white"
                  }`}
                >
                  {index + 1}
                </div>
              </label>
            ))}
          </div>
          <SecondaryButton className="w-fit" onClick={handleSubmit}>
            Submit
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
};

export default Rating;
