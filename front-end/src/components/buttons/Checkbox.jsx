const Checkbox = ({ checked, onChange, label, className = "" }) => {
  return (
    <label className={`flex items-center cursor-pointer ${className}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="hidden"
      />
      <span
        className={`bg-radial from-warmyellow text-lightgray to-electricblue rounded-lg p-2 flex items-center justify-center transition hover:opacity-90`}
      >
        {checked && (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4 text-blueblack"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 13l4 4L19 7"
            />
          </svg>
        )}
      </span>
      {label && (
        <span className="ml-2 text-sm font-semibold text-lightgray">
          {label}
        </span>
      )}
    </label>
  );
};

export default Checkbox;
