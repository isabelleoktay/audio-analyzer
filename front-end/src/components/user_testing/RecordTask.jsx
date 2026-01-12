import SecondaryButton from "../../components/buttons/SecondaryButton.jsx";

const RecordTask = ({ onNext, config }) => {
  const { phase = "pre-practice", label = "", taskIndex } = config || {};
  const isPost = phase === "post-practice";

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h2 className="text-3xl text-electricblue font-bold mb-4 text-center">
        {isPost ? "Final Recording" : "Initial Recording"}
      </h2>
      <p className="text-center mb-6">
        Task {taskIndex !== undefined ? taskIndex + 1 : ""} – {label}
      </p>
      <p className="text-center mb-6">
        {isPost
          ? "Please record again after the practice to see improvements."
          : "Please record once before the practice session."}
      </p>
      <div className="pt-6">
        <SecondaryButton onClick={() => onNext()}>
          Continue to {isPost ? "next step" : "practice"}.
        </SecondaryButton>
      </div>
    </div>
  );
};

export default RecordTask;
