import InstrumentCard from "../components/cards/InstrumentCard.jsx";
import { ReactComponent as ViolinIcon } from "../assets/violin.svg";
import { ReactComponent as VoiceIcon } from "../assets/voice.svg";
import { ReactComponent as PolyphonicIcon } from "../assets/polyphonic.svg";

const instruments = [
  { Icon: ViolinIcon, label: "violin" },
  { Icon: VoiceIcon, label: "voice" },
  { Icon: PolyphonicIcon, label: "polyphonic" },
];

export default function Analyzer() {
  return (
    <div className="flex flex-col items-center justify-center">
      <h2 className="text-xl mb-6 text-lightgray">
        Which instrument are you analyzing today?
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        {instruments.map((inst) => (
          <InstrumentCard key={inst.label} {...inst} />
        ))}
      </div>
    </div>
  );
}
