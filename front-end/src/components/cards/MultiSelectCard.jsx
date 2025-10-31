import { useState, useEffect, useRef } from "react";
import SurveyCheckbox from "../buttons/SurveyCheckbox";

const MultiSelectCard = ({
  question,
  options = [],
  onChange,
  allowOther = true,
  value = [],
  background_colour = "bg-bluegray/25",
  isMultiSelect = true,
  showToggle = false,
  miniVersion = false,
  selected = [],
}) => {
  const containerRef = useRef(null);
  const [columnWidth, setColumnWidth] = useState(150);

  const [selectedOptions, setSelectedOptions] = useState(selected || []);
  const [otherText, setOtherText] = useState("");
  const [multiMode, setMultiMode] = useState(isMultiSelect);

  const firstLoad = useRef(true);

  // Sync external value
  useEffect(() => {
    if (!value || !value.length) return;
    if (firstLoad.current) {
      const regularOptions = value.filter((v) => options.includes(v));
      const otherValue = value.find((v) => !options.includes(v));
      setSelectedOptions([...regularOptions, ...(otherValue ? ["Other"] : [])]);
      setOtherText(otherValue || "");
      firstLoad.current = false;
    }
  }, [value, options]);

  // Measure column width dynamically
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
      const width = tempSpan.getBoundingClientRect().width + 32;
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
    let updated;

    if (multiMode) {
      updated = selectedOptions.includes(option)
        ? selectedOptions.filter((o) => o !== option)
        : [...selectedOptions, option];
    } else {
      updated = selectedOptions.includes(option) ? [] : [option];
    }

    setSelectedOptions(updated);
    triggerChange(updated, otherText);
  };

  const toggleOther = () => {
    const hasOther = selectedOptions.includes("Other");
    let updated;

    if (multiMode) {
      updated = hasOther
        ? selectedOptions.filter((o) => o !== "Other")
        : [...selectedOptions, "Other"];
    } else {
      updated = hasOther ? [] : ["Other"];
    }

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
      className={`${background_colour} ${
        miniVersion
          ? "rounded-lg p-4 flex flex-col items-center w-full"
          : "rounded-3xl p-8 flex flex-col items-center w-full"
      } transition-all duration-200`}
      ref={containerRef}
    >
      {/* Header */}
      <div
        className={`w-full flex justify-between items-center ${
          miniVersion ? "mb-1" : "mb-4"
        }`}
      >
        <h4
          className={`${
            miniVersion
              ? "text-s font-medium text-lightpink text-center flex-1"
              : "text-xl font-semibold text-lightpink text-center flex-1"
          }`}
        >
          {question}
        </h4>

        {showToggle && (
          <button
            onClick={() => setMultiMode((prev) => !prev)}
            className={`${
              miniVersion
                ? "text-[10px] text-white bg-lightpink/40 hover:bg-lightpink/60 rounded-md px-2 py-[1px] ml-2 transition"
                : "text-sm text-white bg-lightpink/40 hover:bg-lightpink/60 rounded-md px-3 py-1 ml-4 transition"
            }`}
          >
            {multiMode ? "Multi Select" : "Single Select"}
          </button>
        )}
      </div>

      {/* Option Grid */}
      <div
        className={`grid ${
          miniVersion ? "gap-1.5" : "gap-4"
        } w-full justify-center`}
        style={{
          gridTemplateColumns: `repeat(auto-fit, minmax(${
            miniVersion ? columnWidth * 0.7 : columnWidth
          }px, 1fr))`,
        }}
      >
        {options.map((opt, index) => (
          <div
            key={index}
            className={`flex flex-col items-start ${
              miniVersion ? "text-[11px]" : "text-sm"
            }`}
          >
            <SurveyCheckbox
              label={opt}
              checked={selectedOptions.includes(opt)}
              onChange={() => toggleOption(opt)}
              miniVersion={miniVersion}
            />
          </div>
        ))}

        {allowOther && (
          <div
            className={`flex flex-col items-start ${
              miniVersion ? "text-[11px]" : "text-sm"
            }`}
          >
            <SurveyCheckbox
              label="Other"
              checked={selectedOptions.includes("Other")}
              onChange={toggleOther}
              miniVersion={miniVersion}
            />
            {selectedOptions.includes("Other") && (
              <input
                type="text"
                value={otherText}
                onChange={handleOtherTextChange}
                placeholder="Please specify"
                className={`mt-1 rounded-md border border-lightgray ${
                  miniVersion ? "p-1 text-[11px]" : "p-2 text-sm"
                } w-full`}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MultiSelectCard;
