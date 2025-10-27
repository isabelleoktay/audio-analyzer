import { useState, useRef, useEffect } from "react";
import SurveyScale from "./SurveyScale";

const SurveyScaleRating = ({
  question,
  options = [],
  scaleLabels = ["1", "2", "3", "4", "5"],
  onChange,
  value,
}) => {
  const [ratings, setRatings] = useState([]);
  const containerRef = useRef(null);
  const [scaleWidth, setScaleWidth] = useState(0);

  useEffect(() => {
    if (value) {
      setRatings(value);
    }
  }, [value]);

  // Dynamically calculate 3/4 of container for scale row
  useEffect(() => {
    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        const totalWidth = containerRef.current.offsetWidth;
        setScaleWidth((totalWidth * 3) / 4);
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  const handleSelect = (option, value) => {
    const updated = { ...ratings, [option]: value };
    setRatings(updated);
    onChange?.(updated);
  };

  return (
    <div
      className="bg-bluegray/25 rounded-3xl p-8 flex flex-col items-center w-full"
      ref={containerRef}
    >
      <h4 className="text-xl font-semibold text-lightpink mb-6 text-center">
        {question}
      </h4>

      <div className="flex flex-col w-full gap-3">
        {/* Options */}
        {options.map((opt, index) => (
          <div key={index} className="flex items-center gap-4 w-full">
            {/* Feature name */}
            <div className="flex-1 text-lightgray font-medium">{opt}</div>

            {/* Scale buttons */}
            <SurveyScale
              scaleLabels={scaleLabels}
              selectedValue={ratings[opt]}
              onSelect={(value) => handleSelect(opt, value)}
              width={scaleWidth}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default SurveyScaleRating;
