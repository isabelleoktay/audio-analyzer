import SecondaryButton from "../../components/buttons/SecondaryButton.jsx";

const RecordTask = ({ onNext, surveyData }) => {
  const handleContinue = () => {
    onNext();
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <div className="pt-10">
        <SecondaryButton onClick={handleContinue}>
          Continue to practice.
        </SecondaryButton>
      </div>
    </div>
  );
};

export default RecordTask;
