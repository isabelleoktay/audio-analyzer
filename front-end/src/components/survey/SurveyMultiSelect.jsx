import { useState, useEffect, useRef } from "react";
import SurveyCheckbox from "../buttons/SurveyCheckbox";

const SurveyMultiSelect = ({ question, options = [], onChange, allowOther = true }) => {
  const [selectedOptions, setSelectedOptions] = useState([]);
  const [otherText, setOtherText] = useState("");
  const [columnWidth, setColumnWidth] = useState(150); // default min
  const containerRef = useRef(null);

  // Measure the longest option text
  useEffect(() => {
    if (!containerRef.current) return;

    const tempSpan = document.createElement("span");
    tempSpan.style.visibility = "hidden";
    tempSpan.style.position = "absolute";
    tempSpan.style.whiteSpace = "nowrap";
    tempSpan.style.fontSize = "14px"; // same as label
    document.body.appendChild(tempSpan);

    let maxWidth = 150;
    options.forEach((opt) => {
      tempSpan.innerText = opt;
      const width = tempSpan.getBoundingClientRect().width + 32; // add checkbox + padding
      if (width > maxWidth) maxWidth = width;
    });
    if (allowOther) {
      tempSpan.innerText = "Other";
      const width = tempSpan.getBoundingClientRect().width + 32;
      if (width > maxWidth) maxWidth = width;
    }

    document.body.removeChild(tempSpan);
    setColumnWidth(maxWidth);
  }, [options, allowOther]);

  const toggleOption = (option) => {
    const updated = selectedOptions.includes(option)
      ? selectedOptions.filter((o) => o !== option)
      : [...selectedOptions, option];
    setSelectedOptions(updated);
    triggerChange(updated, otherText);
  };

  const toggleOther = () => {
    const hasOther = selectedOptions.includes("Other");
    const updated = hasOther ? selectedOptions.filter((o) => o !== "Other") : [...selectedOptions, "Other"];
    if (hasOther) setOtherText("");
    setSelectedOptions(updated);
    triggerChange(updated, otherText);
  };

  const handleOtherTextChange = (e) => {
    const text = e.target.value;
    setOtherText(text);
    triggerChange(selectedOptions, text);
  };

  const triggerChange = (selected, otherVal) => {
    const output = selected.map((o) => (o === "Other" ? otherVal : o));
    onChange?.(output);
  };

  return (
    <div className="bg-bluegray/25 rounded-3xl p-8 flex flex-col items-center" ref={containerRef}>
      <h4 className="text-xl font-semibold text-lightpink mb-6 text-center">{question}</h4>

      <div
        className="grid gap-4 w-full justify-center"
        style={{
          gridTemplateColumns: `repeat(auto-fit, minmax(${columnWidth}px, 1fr))`,
        }}
      >
        {options.map((opt, index) => (
          <div key={index} className="flex flex-col items-start">
            <SurveyCheckbox
              label={opt}
              checked={selectedOptions.includes(opt)}
              onChange={() => toggleOption(opt)}
            />
          </div>
        ))}

        {allowOther && (
          <div className="flex flex-col items-start">
            <SurveyCheckbox
              label="Other"
              checked={selectedOptions.includes("Other")}
              onChange={toggleOther}
            />
            {selectedOptions.includes("Other") && (
              <input
                type="text"
                value={otherText}
                onChange={handleOtherTextChange}
                placeholder="Please specify"
                className="mt-2 rounded-md border border-lightgray p-2 text-sm w-full"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SurveyMultiSelect;
