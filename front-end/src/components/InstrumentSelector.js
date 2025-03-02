import React, { useCallback, useMemo } from "react";
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
  const instrumentOptions = useMemo(
    () =>
      Object.keys(instruments).map((key) => ({
        value: key,
        label: instruments[key].label,
      })),
    [] // Dependency array remains empty because `instruments` is a constant and won't change.
  );

  const handleInstrumentChange = useCallback(
    (selectedInstrument) => {
      const { minNote, maxNote } = instruments[selectedInstrument];
      setMinNote(minNote);
      setMaxNote(maxNote);
    },
    [setMinNote, setMaxNote]
  );

  return (
    <div className="relative w-3/4 flex items-center space-x-4">
      <label className="text-gray-600 whitespace-nowrap">
        2. Select instrument:
      </label>
      <Dropdown options={instrumentOptions} onSelect={handleInstrumentChange} />
    </div>
  );
};

export default InstrumentSelector;
