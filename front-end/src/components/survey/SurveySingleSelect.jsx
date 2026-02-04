import { useState, useEffect } from "react";
import SurveyButton from "../buttons/SurveyButton";

export default function SurveySingleSelect({
  question,
  options = [],
  onChange,
  allowOther = false,
  value = null,
}) {
  const [selected, setSelected] = useState(null);
  const [otherText, setOtherText] = useState("");

  useEffect(() => {
    // If an explicit value is provided, use it
    if (value && !options.includes(value)) {
      // value is an 'Other' free-text value
      setSelected("Other");
      setOtherText(value);
      return;
    }

    if (value) {
      setSelected(value);
      setOtherText("");
      return;
    }

    // No explicit controlled value: reset to null only when question changes
    // Don't reset on every parent render due to options array reference change
    setSelected(null);
    setOtherText("");
  }, [value, question]);

  const handleSelect = (option) => {
    setSelected(option);
    setOtherText("");
    onChange?.(option);
  };

  const toggleOther = () => {
    const isOtherSelected = selected === "Other";
    if (isOtherSelected) {
      setSelected(null);
      setOtherText("");
      onChange?.(null);
    } else {
      setSelected("Other");
      onChange?.(otherText || "");
    }
  };

  const handleOtherTextChange = (e) => {
    const text = e.target.value;
    setOtherText(text);
    setSelected("Other");
    onChange?.(text);
  };

  return (
    <div className="bg-bluegray/25 rounded-3xl p-8">
      <h4 className="text-xl font-semibold text-lightpink mb-6 text-center">
        {question}
      </h4>

      <div className="flex flex-wrap justify-center gap-4 items-center">
        {options.map((opt, index) => (
          <SurveyButton
            key={index}
            onClick={() => handleSelect(opt)}
            isSelected={selected === opt}
          >
            {opt}
          </SurveyButton>
        ))}

        {allowOther && (
          <div className="flex items-center gap-2">
            <SurveyButton
              onClick={toggleOther}
              isSelected={selected === "Other"}
            >
              Other
            </SurveyButton>
            {selected === "Other" && (
              <input
                type="text"
                value={otherText}
                onChange={handleOtherTextChange}
                className="p-2 rounded-md border border-gray-300 w-28 text-center text-sm focus:outline-none focus:ring-2 focus:ring-lightpink"
                placeholder="Type here"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
