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
                Singing is central to music, yet few systems provide
                interactive, pedagogically meaningful feedback on vocal
                technique. We present MuSA Voice, an open web-based platform for
                analysing singing performance through interpretable visual
                feedback. The system provides time-aligned visual feedback on
                vocal technique, powered by pretrained audio models (CLAP and
                Whisper) with specialised classifiers for pitch- and
                timbre-related vocal techniques. These models support analysis
                of technique usage beyond pitch, providing an external
                representation of the voice for reflective learning. Users can
                inspect technique predictions over time, compare segments, and
                engage with visualisations designed to support discussion and
                self-assessment. MuSA demonstrates how pretrained audio models
                can be embedded into an open web platform for interactive vocal
                technique feedback.
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
      <div className="w-full flex justify-center">
        {/* MuSA Voice info */}
        <div className="w-full flex bg-bluegray/25 rounded-3xl">
          {/* LEFT */}
          <div className="flex-1 flex items-center justify-center p-10">
            <div className="w-full">
              <img
                src="/images/musa_img.png"
                alt="musa_img"
                className="w-full rounded-lg"
              />
            </div>
          </div>

          {/* RIGHT */}
          <div className="flex-1 flex items-center justify-center p-12">
            <div className="max-w-xl space-y-5">
              <h1 className="text-4xl font-bold text-lightpink">
                Performance Analysis.
              </h1>
              <p className="text-lg text-lightgray">
                MuSA is a technology-enhanced learning (TEL) tool designed to
                improve music practice by addressing a critical, often-neglected
                component of skill develop- ment: the reflection phase. MuSA is
                a pedagogically- grounded platform for analyzing recorded
                performances to make reflection more efficient and effective.
                MuSA’s design is informed by key educational theories, in-
                cluding the Talent-Development-in-Achievement-Domains (TAD)
                Music Model and learner-centered teaching (LCT) principles like
                scaffolding, self-regulated learning (SRL), and self-directed
                learning (SDL).
              </p>
              <p className="text-lg text-lightgray">
                The performance analyzer is the original and core component of
                MuSA. Its central feature is saliency analysis, which
                algorithmically identifies key moments in a performance based on
                variability in musical features such as pitch, dynamics, and
                tempo. The performance analyzer was the foundation for MuSA
                Voice, into which current work focuses on. The original
                performance analyzer tool can be accessed in the navigation bar.
              </p>
              <p className="text-lg text-lightgray">
                Unlike tools that offer prescriptive, "correct/incorrect"
                feedback, MuSA encourages a learner’s own inter- pretation. As
                an accessible, web-based platform, it allows users to upload or
                record audio for analysis, supporting reflective learning in
                between in-person music learning classes and independent
                practice.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* SMALL CONTROLLED GAP */}
      <div className="h-8" />

      {/* Links */}
      <div className="w-full flex justify-center bg-bluegray/25 rounded-3xl">
        <div className="flex flex-col items-center p-12 w-full max-w-3xl">
          <h1 className="text-4xl font-bold text-lightpink text-center">
            Links for more.
          </h1>
          <div className="w-full flex flex-wrap justify-center mt-6 gap-8">
            <SecondaryButton
              onClick={() => {
                window.open("https://github.com/isabelleoktay/audio-analyzer");
              }}
              fromColor="from-lightpink/80"
              toColor="to-darkpink/80"
              hoverFromColor="from-lightpink"
              hoverToColor="to-darkpink"
            >
              MuSA GitHub
            </SecondaryButton>
            <SecondaryButton
              onClick={() => {
                window.open("https://appskynote.com/research");
              }}
              fromColor="from-lightpink/80"
              toColor="to-darkpink/80"
              hoverFromColor="from-lightpink"
              hoverToColor="to-darkpink"
            >
              skynote project
            </SecondaryButton>
            <SecondaryButton
              onClick={() => {
                window.open(
                  "https://analyzer.appskynote.com/musavoice-testing",
                );
              }}
              fromColor="from-lightpink/80"
              toColor="to-darkpink/80"
              hoverFromColor="from-lightpink"
              hoverToColor="to-darkpink"
              disabled={true}
            >
              join our user test (now closed)
            </SecondaryButton>
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
