import SecondaryButton from "../../components/buttons/SecondaryButton.jsx";

const RecordTask = ({ currentTask = "Pitch Modulation Control" }) => {
  const onClick = () => {
    // logic to start the testing procedure
    window.location.href = "/musavoice-testing-practice";
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <div className="pt-10">
        <SecondaryButton onClick={() => onClick()}>
          Continue to practice.
        </SecondaryButton>
      </div>
    </div>
  );
};

export default RecordTask;
