import SecondaryButton from "../../components/buttons/SecondaryButton.jsx";

const Practice = ({ onNext, config }) => {
  const {
    condition = "control",
    usesTool = false,
    label = "",
    taskIndex,
  } = config || {};

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h2 className="text-3xl text-electricblue font-bold mb-4 text-center">
        Practice Session
      </h2>
      <p className="text-center mb-6">
        Task {taskIndex !== undefined ? taskIndex + 1 : ""} – {label}
      </p>
      <p className="text-center mb-6">
        {usesTool
          ? "Use the feedback tool during this practice."
          : "Do NOT use the feedback tool during this practice (control)."}
      </p>
      <div className="pt-6">
        <SecondaryButton
          onClick={() =>
            onNext({
              lastPracticeCondition: condition,
              lastPracticeUsesTool: usesTool,
              lastPracticeTaskIndex: taskIndex,
            })
          }
        >
          Continue.
        </SecondaryButton>
      </div>
    </div>
  );
};

export default Practice;
