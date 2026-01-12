import SecondaryButton from "../buttons/SecondaryButton";

const ConsentCardNew = ({ handleClick, config = {} }) => {
  const { title = "", textparts = [], buttons = [] } = config;

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8 text-center">
        {config.title}
      </h1>
      <div className="text-justify w-full md:w-1/2">
        {textparts.map((para, i) => (
          <p key={i} className="mb-6">
            {para.map((seg, j) =>
              seg.bold ? (
                <span key={j} className="font-bold">
                  {seg.text}
                </span>
              ) : (
                <span key={j}>{seg.text}</span>
              )
            )}
          </p>
        ))}
      </div>
      <div className="flex gap-4">
        <SecondaryButton onClick={() => handleClick(true)}>
          {config.buttonLabels[0]}
        </SecondaryButton>
        <SecondaryButton onClick={() => handleClick(false)}>
          {config.buttonLabels[1]}
        </SecondaryButton>
      </div>
    </div>
  );
};

export default ConsentCardNew;
