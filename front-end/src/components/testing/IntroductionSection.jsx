import { useState } from "react";
import SecondaryButton from "../buttons/SecondaryButton";

const IntroductionSection = ({ handleNextStep, subjectData }) => {
  const [musicExperience, setMusicExperience] = useState("");

  const handleSelectionChange = (event) => {
    setMusicExperience(event.target.value);
    subjectData.musicExperience = event.target.value; // Save the selection to subjectData
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray mt-20 md:mt-0 mb-28">
      <h1 className="text-5xl text-electricblue font-bold mb-8">
        Introduction{" "}
      </h1>
      <div className="flex flex-col items-center text-sm md:text-base justify-center w-full md:w-1/2 text-justify">
        <p className="mb-6 font-bold">
          Thank you for participating in this study! This experiment is designed
          to explore how visual feedback of audio features may support the
          improvement of music performance.
        </p>
        <p className="mb-6">
          You will be randomly assigned to start with one of two rounds of the
          experiment, both of which involve recording yourself singing short
          musical phrases under specific conditions. You will be asked to
          complete both rounds of the experiment.
        </p>
        <p className="mb-6 font-bold">
          Before you begin, please answer the following question to help us
          understand your music background:
        </p>
        <div className="mb-6 bg-blueblack/25 p-6 rounded-3xl w-full">
          <p className="font-bold text-lg md:text-2xl mb-4 text-lightpink text-left md:text-center">
            How much musical experience do you have?
          </p>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-start">
            {["Beginner", "Intermediate", "Advanced", "Professional"].map(
              (level) => (
                <label
                  key={level}
                  className={`grid grid-cols-3 md:grid-cols-1 gap-4 items-start cursor-pointer ${
                    musicExperience === level
                      ? "text-lightpink"
                      : "text-gray-500"
                  }`}
                >
                  <input
                    type="radio"
                    value={level}
                    checked={musicExperience === level}
                    onChange={handleSelectionChange}
                    className="hidden"
                  />
                  <div
                    className={`col-span-1 p-2 flex items-center justify-center rounded-full border-2 ${
                      musicExperience === level
                        ? "border-lightpink bg-darkpink text-white"
                        : "border-gray-300 bg-white hover:border-lightpink hover:bg-lightpink hover:text-white"
                    }`}
                  >
                    {level}
                  </div>
                  <span className="col-span-2 text-left text-sm self-start text-lightgray">
                    {level === "Beginner"
                      ? "Learning the basics through lessons or self-teaching"
                      : level === "Intermediate"
                      ? "Applying foundational skills consistently in practice and performances"
                      : level === "Advanced"
                      ? "Employing high technical proficiency with artistic expression; teaching and composing"
                      : "Innovating in the field through performances, recordings, or leadership"}
                  </span>
                </label>
              )
            )}
          </div>
        </div>
      </div>
      <SecondaryButton onClick={() => handleNextStep("instructions")}>
        Continue
      </SecondaryButton>
    </div>
  );
};

export default IntroductionSection;
