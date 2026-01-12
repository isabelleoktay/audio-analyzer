import SecondaryButton from "../buttons/SecondaryButton";

const ConsentCard = ({
  handleConsent,
  title = "Welcome to the Audio Analyzer Test",
}) => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-lightgray">
      <h1 className="text-5xl text-electricblue font-bold mb-8 text-center">
        {title}
      </h1>
      <div className="text-justify w-full md:w-1/2">
        <p className="mb-6">
          By participating in this study, you acknowledge that audio recordings
          will be collected and used for educational and scientific research
          purposes.
        </p>
        <p className="mb-6">
          The data gathered will be limited to the audio itself and computed
          features derived from the recordings. These features may include, but
          are not limited to, vocal technique, dynamics, pitch, vibrato, and
          tempo. All data will be stored securely and used solely for research
          and educational analysis, with no commercial intent.
        </p>
        <p className="mb-6">
          <span className="font-bold">
            No personal or identifying information will be collected.
          </span>{" "}
          Participation is voluntary, and you may withdraw at any time without
          consequence.
        </p>
        <p className="mb-6">Do you consent to participate in this study?</p>
      </div>
      <div className="flex gap-4">
        <SecondaryButton onClick={() => handleConsent(true)}>
          Yes, I Consent
        </SecondaryButton>
        <SecondaryButton onClick={() => handleConsent(false)}>
          No, Take Me Back
        </SecondaryButton>
      </div>
    </div>
  );
};

export default ConsentCard;
