import React, { useState } from "react";
import InstrumentSelector from "../InstrumentSelector";
import "./NoteRangeSelector.css";

const NoteRangeSelector = ({ minNote, maxNote, setMinNote, setMaxNote }) => {
  const [showNoteRange, setShowNoteRange] = useState(false);

  const handleToggleNoteRange = () => {
    setShowNoteRange(!showNoteRange);
  };

  const handleMinNoteChange = (event) => {
    setMinNote(event.target.value);
  };

  const handleMaxNoteChange = (event) => {
    setMaxNote(event.target.value);
  };

  return (
    <div className="w-full flex flex-col items-center">
      <InstrumentSelector setMinNote={setMinNote} setMaxNote={setMaxNote} />
      <div className="text-gray-500 text-sm my-2">
        Or{" "}
        <span
          className="underline hover:cursor-pointer hover:text-blue-500"
          onClick={handleToggleNoteRange}
        >
          specify note range
        </span>
      </div>
      <div
        className={`note-range-container ${showNoteRange ? "show" : "hide"}`}
      >
        <label className="block text-gray-600">
          Min Note:
          <input
            type="text"
            value={minNote}
            onChange={handleMinNoteChange}
            className="ml-2 p-1 border rounded"
            placeholder="e.g., F3"
          />
        </label>
        <label className="block text-gray-600 mt-2">
          Max Note:
          <input
            type="text"
            value={maxNote}
            onChange={handleMaxNoteChange}
            className="ml-2 p-1 border rounded"
            placeholder="e.g., B6"
          />
        </label>
      </div>
    </div>
  );
};

export default NoteRangeSelector;
