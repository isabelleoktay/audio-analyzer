const SurveyCheckbox = ({ label, checked, onChange }) => {
  return (
    <label className="flex items-center cursor-pointer select-none gap-2">
      <div className="relative">
        {/* Hidden native checkbox for state */}
        <input
          type="checkbox"
          checked={checked}
          onChange={onChange}
          className="absolute opacity-0 w-6 h-6 cursor-pointer"
        />

        {/* Styled square */}
        <div
          className={`
            w-6 h-6 rounded-md flex items-center justify-center
            transition-colors duration-200
            ${checked ? "bg-darkpink border-darkpink" : "bg-lightgray border-2 border-lightgray"}
            hover:bg-lightpink
          `}
        >
          {/* Tick always visible */}
          <svg
            className="w-3 h-3 text-white"
            fill="none"
            stroke="currentColor"
            strokeWidth="4"
            viewBox="0 0 24 24"
          >
            <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>

      {/* Label */}
      <span className="text-lightgray text-sm hover:text-lightpink">{label}</span>
    </label>
  );
};

export default SurveyCheckbox;
