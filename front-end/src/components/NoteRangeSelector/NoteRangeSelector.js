import React, { useState } from "react";
import InstrumentSelector from "../InstrumentSelector";
import "./NoteRangeSelector.css";
import SmallTextLink from "../SmallTextLink";

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
    <div className="w-full flex flex-col items-center overflow-visible">
      <InstrumentSelector setMinNote={setMinNote} setMaxNote={setMaxNote} />
      <SmallTextLink
        nonLinkText="Or "
        linkText="specify note range"
        handleClick={handleToggleNoteRange}
      />
      <div
        className={`note-range-container ${showNoteRange ? "show" : "hide"}`}
      >
        <label className="block text-gray-600">
          Min Note:
          <input
            type="text"
            value={minNote}
            onChange={handleMinNoteChange}
            className="ml-2 p-1 border rounded shadow-sm"
            placeholder="e.g., F3"
          />
        </label>
        <label className="block text-gray-600 mt-2">
          Max Note:
          <input
            type="text"
            value={maxNote}
            onChange={handleMaxNoteChange}
            className="ml-2 p-1 border rounded shadow-sm"
            placeholder="e.g., B6"
          />
        </label>
      </div>
    </div>
  );
};

export default NoteRangeSelector;
