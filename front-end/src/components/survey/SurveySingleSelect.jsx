import { useState } from "react";
import SurveyButton from "../buttons/SurveyButton";

export default function SurveySingleSelect({ question, options = [], onChange }) {
  const [selected, setSelected] = useState(null);

  const handleSelect = (option) => {
    setSelected(option);
    onChange?.(option); // use onChange instead of onSelect
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
            isSelected={selected === opt}
          >
            {opt}
          </SurveyButton>
        ))}
      </div>
    </div>
  );
}