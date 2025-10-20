import { useState } from "react";
import SurveyButton from "../buttons/SurveyButton";

export default function SurveySingleSelect({ question, options = [], onSelect }) {
  const [selected, setSelected] = useState(null); // <-- rename correctly

  const handleSelect = (option) => {
    setSelected(option);
    onSelect?.(option);
  };

  return (
    <div className="bg-bluegray/25 rounded-3xl p-8">
      <h4 className="text-xl font-semibold text-lightpink mb-6 text-center">
        {question}
      </h4>

      <div className="flex flex-wrap justify-center gap-6">
        {options.map((opt, index) => (
          <SurveyButton
            key={index}
            onClick={() => handleSelect(opt)}
            isActive={true}
            isSelected={selected === opt} // <-- compare state value
          >
            {opt}
          </SurveyButton>
        ))}
      </div>
    </div>
  );
}
