import { useState } from "react";
import SecondaryButton from "../buttons/SecondaryButton";

const Questionnaire = ({ onSubmit }) => {
  const [answers, setAnswers] = useState({
    experience: "",
    improvement: "",
    payment: "",
  });

  const handleChange = (event) => {
    setAnswers((prev) => ({
      ...prev,
      [event.target.name]: event.target.value,
    }));
  };

  const handleSubmit = () => {
    if (!answers.experience || !answers.payment || !answers.improvement) {
      alert("Please answer all questions before submitting.");
      return;
    }
    onSubmit(answers);
  };

  return (
    <div className="flex flex-col items-center justify-center text-center min-h-screen text-lightgray w-full md:w-1/2 gap-4 md:gap-6">
      <div className="flex flex-col mb-8 space-y-8">
        <div className="text-4xl text-electricblue font-bold">
          Final Questionnaire
        </div>
        <div className="flex flex-col items-center justify-center w-full space-y-6 bg-blueblack/25 p-6 rounded-3xl">
          <div className="flex flex-col space-y-6 w-full text-left text-sm md:text-lg">
            <label className="text-lightgray">
              How was your experience with the audio analyzer? What did you like
              or dislike about it? How did you feel about your music performance
              in using versus not using the audio analyzer?
              <textarea
                name="experience"
                value={answers.experience}
                onChange={handleChange}
                className="w-full p-2 mt-2 md:mt-4 rounded-lg border-none bg-blueblack/50 focus:outline-none"
                rows="4"
              />
            </label>
            <label className="text-lightgray">
              Do you think the audio analyzer helped improve your performances
              of the reference audios? If so, how? If not, why not?
              <textarea
                name="improvement"
                value={answers.improvement}
                onChange={handleChange}
                className="w-full p-2 mt-2 md:mt-4 rounded-lg border-none bg-blueblack/50 focus:outline-none"
                rows="4"
              />
            </label>
            <label className="text-lightgray">
              Would you pay for a similar audio analysis service? If so, how
              much?
              <textarea
                name="payment"
                value={answers.payment}
                onChange={handleChange}
                className="w-full p-2 mt-2 md:mt-4 rounded-lg border-none bg-blueblack/50 focus:outline-none"
                rows="4"
              />
            </label>
          </div>
          <SecondaryButton className="w-fit" onClick={handleSubmit}>
            Submit
          </SecondaryButton>
        </div>
      </div>
    </div>
  );
};

export default Questionnaire;
