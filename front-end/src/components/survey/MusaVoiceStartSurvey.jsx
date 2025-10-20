import SurveySingleSelect from "./SurveySingleSelect.jsx";
import SurveyMultiSelect from "./SurveyMultiSelect.jsx";
import SurveyScaleRating from "./SurveyScaleRating.jsx";
import SecondaryButton from "../buttons/SecondaryButton.jsx";


const MusaVoice = () => {
  const handleAnswer = (answer) => {
    console.log("User selected:", answer);
  };

  const handleContinue = () => {
    console.log("Continuing to MuSA voice tool...");
  };

  return (
    <div className="min-h-screen p-12 pt-20">
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Question 1 */}
        <SurveySingleSelect
          question="How much singing experience do you have?"
          options={["Beginner", "Intermediate", "Advanced", "Professional"]}
          onSelect={handleAnswer}
        />
        {/* Question 2 */}
        <SurveyMultiSelect
          question="How do you typically practice?"
          options={["In person vocal coaching lessons", "Group singing e.g. choir, quartet", 
            "Independent practice e.g. at home", "With a band(s)", "Recording videos/audio of myself",]}
          onChange={(selected) => console.log("Selected instruments:", selected)}
        />
        {/* Question 3 */}
        <SurveySingleSelect
          question="How many hours do you practice per week?"
          options={["< 1h", "1-2h", "2-5h", "5-10h", "10-15h", "> 15h"]}
          onSelect={handleAnswer}
        />

        {/* Question 4 */}
        <SurveySingleSelect
          question="Have you used automatic vocal analysis or visualiser tools before?"
          options={["Yes", "No", "I'm not sure"]}
          onSelect={handleAnswer}
        />

        {/* Question 5 */}
        <SurveyMultiSelect
          question="What genres/styles do you typically sing?"
          options={["Pop", "Rock", "RnB", "Soul", "Classical/Opera", "Jazz", "Musical Theatre", "Folk", "Country", "Hip Hop", "Heavy metal"]}
          onChange={(selected) => console.log("Selected instruments:", selected)}
        />

        {/* Question 6 */}
        <SurveySingleSelect
          question="What is your primary voice type?"
          options={["Bass", "Baritone", "Tenor", "Alto", "Mezzo", "Soprano", "I'm not sure"]}
          onSelect={handleAnswer}
        />

        {/* Question 7 */}
        <SurveyScaleRating
          question="Rate how useful it would be for you to get feedback (presence/absence of them) on the following vocal techniques:"
          options={["Vibrato", "Trill", "Trillo", "Straight tone", "Belting", "Breathiness", "Speech tone", "Lip trill", "Vocal fry", 
            "PortaInhaled singing", "Glissando", "Mixed voice", "Head voice", "Falsetto", "Pharyngeal" ]}
          scaleLabels={["Not useful", "", "", "", "Very useful"]}
          onChange={(ratings) => console.log(ratings)}
        />

        {/* Question 8 */}
        <SurveySingleSelect
          question="Will you be in a quiet room while using the tool?"
          options={["Yes", "No"]}
          onSelect={handleAnswer}
        />

        {/* Question 9 */}
        <SurveySingleSelect
          question="Are you using headphones while using the tool?"
          options={["Yes", "No"]}
          onSelect={handleAnswer}
        />

        {/* Submit answers */}
        <div className="flex justify-center mt-8 w-full">
          <SecondaryButton onClick={handleContinue}>Continue to use MuSA voice</SecondaryButton>
        </div>
      </div>
    </div>
  );
};

export default MusaVoice;
