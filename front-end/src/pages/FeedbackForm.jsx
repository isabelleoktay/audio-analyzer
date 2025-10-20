import { useState, useEffect } from "react";
import SurveySingleSelect from "../components/survey/SurveySingleSelect";
import SurveyMultiSelect from "../components/survey/SurveyMultiSelect";
import SurveyScaleRating from "../components/survey/SurveyMultiScale.jsx";
import SecondaryButton from "../components/buttons/SecondaryButton.jsx";

const FeedbackForm = () => {
  const [showIntro, setShowIntro] = useState(true);

  return (
    <div className="flex items-center justify-center min-h-screen">
        {/* first section of feedback form with button to click to move on */}

        {/* second section of feedback form with button to click to move on */}

        {/* third section of feedback form with button to click to move on */}

        {/* fourth section of feedback form with button to click to move on */}
        
    </div>
  );
};

export default FeedbackForm;
