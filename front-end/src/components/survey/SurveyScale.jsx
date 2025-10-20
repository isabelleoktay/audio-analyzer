const SurveyScale = ({
  scaleLabels = ["1", "2", "3", "4", "5"],
  selectedValue = null,
  onSelect,
  width = "100%",
  isActive = true,
}) => {
  return (
    <div className="flex gap-2" style={{ width }}>
      {scaleLabels.map((label, i) => {
        const value = i + 1;
        const isSelected = selectedValue === value;
        return (
          <button
            key={i}
            onClick={() => isActive && onSelect?.(value)}
            className={`
              flex-1 h-10 rounded-md font-semibold transition-colors
              relative flex items-center justify-center
              ${isSelected ? "bg-darkpink text-white" : "bg-lightgray text-blueblack hover:bg-lightpink"}
            `}
          >
            <span className="absolute text-blueblack">{label}</span>
          </button>
        );
      })}
    </div>
  );
};

export default SurveyScale;
