import SurveyTextAnswer from "../../components/survey/SurveyTextAnswer";
import SecondaryButton from "../../components/buttons/SecondaryButton";

const ThankYou = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray px-4">
      <h1 className="text-5xl text-electricblue font-bold mb-8 text-center">
        Thank you for completing the MuSA vocal technique user test!
      </h1>
      <p className="md:w-3/4 text-justify-center pt-5 pb-5">
        We are extremely grateful for your participation and time. If you are
        interested in our future work and participating in upcoming Vocal
        Performance Technology studies (e.g. comparing different models), 
        we invite you to provide your email below.
      </p>
      <div className="w-full mt-6">
        <SurveyTextAnswer
          question="Provide your email if you are interested in future vocal performance technology work (optional):"
          placeholder=""
        />
      </div>
      <SecondaryButton
        className="mt-10"
        onClick={() => {
          // logic to finish the testing procedure
          window.location.href = "/";
        }}
      >
        finish and exit testing platform
      </SecondaryButton>
    </div>
  );
};

export default ThankYou;
