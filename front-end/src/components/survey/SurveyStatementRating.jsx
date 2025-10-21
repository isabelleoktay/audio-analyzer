import { useState, useRef, useEffect } from "react";
import SurveyScale from "./SurveyScale";

const SurveyStatementRating = ({
  question,
  statements = [],
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

  useEffect(() => {
    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        const totalWidth = containerRef.current.offsetWidth;
        setScaleWidth(totalWidth * 0.9);
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
      {/* Options */}
      {statements.map((statement, index) => (
        <div
          key={index}
          className="w-full flex flex-col gap-2 mb-10 items-center"
        >
          {/* Statement text above the scale */}
          <h5 className="text-l text-lightgray text-center">{statement}</h5>

          {/* Scale buttons */}
          <SurveyScale
            scaleLabels={scaleLabels}
            selectedValue={ratings[statement]}
            onSelect={(value) => handleSelect(statement, value)}
            width={scaleWidth}
          />
        </div>
      ))}
    </div>
  );
};

export default SurveyStatementRating;
