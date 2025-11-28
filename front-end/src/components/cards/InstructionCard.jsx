const InstructionCard = ({ title, description, className = "" }) => {
  return (
    <div
      className={`flex flex-col gap-1 p-6 bg-blueblack/30 border border-1 border-electricblue rounded-2xl ${className}`}
    >
      <div className="text-3xl text-electricblue font-bold">{title}</div>
      <div className="text-lg text-lightgray">{description}</div>
    </div>
  );
};

export default InstructionCard;
