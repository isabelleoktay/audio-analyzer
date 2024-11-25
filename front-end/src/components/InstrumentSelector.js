import React from "react";
import Dropdown from "./Dropdown";

const instruments = {
  violin: { minNote: "G3", maxNote: "A7", label: "Violin" },
  viola: { minNote: "C3", maxNote: "A6", label: "Viola" },
  cello: { minNote: "C2", maxNote: "A5", label: "Cello" },
  trumpet: { minNote: "F#3", maxNote: "C6", label: "Trumpet" },
  trombone: { minNote: "E2", maxNote: "B4", label: "Trombone" },
  clarinet: { minNote: "E3", maxNote: "C6", label: "Clarinet" },
  voiceFemale: { minNote: "C3", maxNote: "C7", label: "Female Voice" },
  voiceMale: { minNote: "C2", maxNote: "G5", label: "Male Voice" },
};

const InstrumentSelector = ({ setMinNote, setMaxNote }) => {
  const handleInstrumentChange = (selectedInstrument) => {
    const { minNote, maxNote } = instruments[selectedInstrument];
    setMinNote(minNote);
    setMaxNote(maxNote);
  };

  const instrumentOptions = Object.keys(instruments).map((key) => ({
    value: key,
    label: instruments[key].label,
  }));

  return (
    <div className="flex justify-center my-4">
      <div className="w-1/4">
        <label className="block text-gray-600 mb-1">
          Select an instrument:
        </label>
        <Dropdown
          options={instrumentOptions}
          onSelect={handleInstrumentChange}
        />
      </div>
    </div>
  );
};

export default InstrumentSelector;
