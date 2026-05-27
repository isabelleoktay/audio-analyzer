import SurveyTextAnswer from "../components/survey/SurveyTextAnswer";
import SecondaryButton from "../components/buttons/SecondaryButton";

const LandingPage = () => {
return (
  <div className="min-h-screen justify-center pt-20 flex flex-col">

    <div className="w-full flex justify-center">
        {/* MuSA Voice info */}
      <div className="w-full flex bg-bluegray/25 rounded-3xl">

        {/* LEFT */}
        <div className="flex-1 flex items-center justify-center p-12">
          <div className="max-w-xl space-y-5">
            <h1 className="text-4xl font-bold text-lightpink">
              Explore the power of vocal analysis with MuSA Voice.
            </h1>
            <p className="text-lg text-lightgray">
              Singing is central to music, yet few systems provide interactive, pedagogically meaningful feedback on vocal technique. 
              We present MuSA Voice, an open web-based platform for analysing singing performance through interpretable visual feedback.
              The system provides time-aligned visual feedback on vocal technique, powered by pretrained audio models 
               (CLAP and Whisper) with specialised classifiers for pitch- and timbre-related vocal techniques. 
               These models support analysis of technique usage beyond pitch, providing an external representation of the voice 
               for reflective learning. Users can inspect technique predictions over time, compare segments, and engage with 
               visualisations designed to support discussion and self-assessment. MuSA demonstrates how pretrained audio models 
               can be embedded into an open web platform for interactive vocal technique feedback.
            </p>
          </div>
        </div>

        {/* RIGHT */}
        <div className="flex-1 flex items-center justify-center p-10">
          <div className="w-full">
            <video className="w-full rounded-lg" controls>
              <source src="video/MuSA demo.mp4" type="video/mp4" />
            </video>
          </div>
        </div>

      </div>
    </div>


    {/* SMALL CONTROLLED GAP */}
    <div className="h-8" />

    {/* ANALYZER */}
    <div className="w-full flex bg-bluegray/25 rounded-3xl">
        <div className="flex-1 flex items-center justify-center p-12">
            <div className="max-w-xl space-y-5">
            <h1 className="text-4xl font-bold text-lightpink">
                Performance Analysis.
            </h1>
            <p className="text-lg text-lightgray">

            </p>
            </div>
        </div>
    </div>

    {/* SMALL CONTROLLED GAP */}
    <div className="h-8" />

    {/* SURVEY SECTION (FULL WIDTH) */}
    <div className="w-full pb-12">

      <div className="w-full">
        <SurveyTextAnswer
          question="We are constantly evolving and looking for input. You can provide your email if you are interested in receiving updates and e.g. participating in user studies of future versions of MuSA voice:"
          placeholder=""
        />
      </div>

      <div className="w-full flex justify-end mt-6">
        <SecondaryButton
          onClick={() => {
            // need to add email submit handling -- remove written text and send to a MongoDB database just for interested peoples emails
          }}
        >
          submit email
        </SecondaryButton>
      </div>
    </div>
  </div>
);
};

export default LandingPage;
