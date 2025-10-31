import { useState, useEffect, useRef } from "react";
import SurveyCheckbox from "../buttons/SurveyCheckbox";

const MultiSelectCard = ({
  question,
  options = [],
  onChange,
  allowOther = true,
  value = [],
  background_colour = "bg-bluegray/25",
}) => {
  const containerRef = useRef(null);
  const [columnWidth, setColumnWidth] = useState(150);

  const [selectedOptions, setSelectedOptions] = useState([]);
  const [otherText, setOtherText] = useState("");

  // Track first load to avoid overwriting user input
  const firstLoad = useRef(true);

  useEffect(() => {
    // Only update state on first load, or if value changed externally
    if (!value || !value.length) return;

    if (firstLoad.current) {
      const regularOptions = value.filter((v) => options.includes(v));
      const otherValue = value.find((v) => !options.includes(v));
      setSelectedOptions([...regularOptions, ...(otherValue ? ["Other"] : [])]);
      setOtherText(otherValue || "");
      firstLoad.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, options]);

  // Measure longest option for responsive layout
  useEffect(() => {
    if (!containerRef.current) return;

    const tempSpan = document.createElement("span");
    tempSpan.style.visibility = "hidden";
    tempSpan.style.position = "absolute";
    tempSpan.style.whiteSpace = "nowrap";
    tempSpan.style.fontSize = "14px";
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
    const updated = hasOther
      ? selectedOptions.filter((o) => o !== "Other")
      : [...selectedOptions, "Other"];
    if (hasOther) setOtherText("");
    setSelectedOptions(updated);
    triggerChange(updated, hasOther ? "" : otherText);
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
    <div
      className={`${background_colour} rounded-3xl p-8 flex flex-col items-center`}
      ref={containerRef}
    >
      <h4 className="text-xl font-semibold text-lightpink mb-6 text-center">
        {question}
      </h4>

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

export default MultiSelectCard;
