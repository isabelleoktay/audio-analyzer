import { useState, useEffect } from "react";

export default function SurveyTextAnswer({ question, onChange, value }) {
  const [text, setText] = useState("");

  useEffect(() => {
    setText(value || "");
  }, [value]);

  const handleChange = (e) => {
    setText(e.target.value);
    onChange?.(e.target.value);
  };

  return (
    <div className="bg-bluegray/25 rounded-3xl p-8 w-full">
      {/* Question with thin underline */}
      <h4 className="text-xl font-semibold text-lightpink mb-4 text-center border-b border-lightpink pb-2">
        {question}
      </h4>

      {/* Input field */}
      <input
        type="text"
        value={text}
        onChange={handleChange}
        placeholder="Add your answer here..."
        className="
          w-full text-lightgray text-sm placeholder-lightgray
          bg-bluegray/0 border-none focus:ring-0 focus:outline-none
          mt-4
        "
      />
    </div>
  );
}
