import SecondaryButton from "../buttons/SecondaryButton";

const IntroductionSection = ({ handleNextStep }) => {
  return (
    <div className="flex flex-col items-center justify-center h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8">
        Introduction{" "}
      </h1>
      <div className="flex flex-col items-center justify-center w-1/2 text-justify">
        <p className="mb-6 ">
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
      </div>
      <SecondaryButton onClick={() => handleNextStep("instructions")}>
        Continue
      </SecondaryButton>
    </div>
  );
};

export default IntroductionSection;
