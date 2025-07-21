const FeatureInstructionsList = () => {
  return (
    <ol className="flex flex-col space-y-2 text-lightpink ml-0">
      <li className="justify-start">
        <span className="font-bold text-darkpink text-lg">Pitch.</span>{" "}
        <span className="font-normal text-lightgray">
          Sing the musical phrase as closely as possible to the pitch of the
          reference audio.
        </span>
      </li>
      <li className="justify-start">
        <span className="font-bold text-darkpink text-lg">Dynamics.</span>{" "}
        <span className="font-normal text-lightgray">
          Sing the musical phrase following the loudness levels of the reference
          audio.
        </span>
      </li>
      <li className="justify-start">
        <span className="font-bold text-darkpink text-lg">Tempo.</span>{" "}
        <span className="font-normal text-lightgray">
          Sing the musical phrase following the tempo changes of the reference
          audio.
        </span>
      </li>
    </ol>
  );
};

export default FeatureInstructionsList;
