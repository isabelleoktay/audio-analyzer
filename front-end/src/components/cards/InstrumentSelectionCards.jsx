import InstrumentCard from "./InstrumentCard.jsx";

const InstrumentSelectionCards = ({ instruments, handleInstrumentSelect }) => {
  return (
    <div className="mt-64 items-center justify-items-center">
      <h2 className="text-xl mb-6 text-lightgray tracking-widest">
        Which instrument are you analyzing today?
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        {instruments.map((inst) => (
          <InstrumentCard
            key={inst.label}
            {...inst}
            onSelect={() => handleInstrumentSelect(inst.label)}
          />
        ))}
      </div>
    </div>
  );
};

export default InstrumentSelectionCards;
