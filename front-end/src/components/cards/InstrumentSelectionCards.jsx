import InstrumentCard from "./InstrumentCard.jsx";

const InstrumentSelectionCards = ({ instruments, handleInstrumentSelect }) => {
  return (
    <div className="mt-20 mb-6 lg:mt-64 items-center justify-items-center">
      <h2 className="text-xl mb-6 text-lightgray text-center tracking-widest">
        Which instrument are you analyzing today?
      </h2>
      <div className="flex lg:flex-row flex-col gap-6 items-center justify-center">
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
