const activeAnalysisButtonClassName =
  "bg-radial from-warmyellow to-darkpink text-blueblack";
const inactiveAnalysisButtonClassName =
  "bg-lightgray/25 text-lightgray hover:bg-radial hover:from-warmyellow/50 hover:to-darkpink/50";

export const analysisButtonConfig = {
  violin: [
    {
      type: "left",
      label: "dynamics",
    },
    { type: "center", label: "pitch" },
    { type: "right", label: "tempo" },
  ],
  voice: [
    { type: "left", label: "dynamics" },
    { type: "center", label: "pitch" },
    { type: "center", label: "phonation" },
    { type: "center", label: "vibrato" },
    { type: "right", label: "tempo" },
  ],
  polyphonic: [
    { type: "left", label: "dynamics" },
    { type: "right", label: "tempo" },
  ],
};

export const analysisButtonClassNames = {
  active: activeAnalysisButtonClassName,
  inactive: inactiveAnalysisButtonClassName,
};
