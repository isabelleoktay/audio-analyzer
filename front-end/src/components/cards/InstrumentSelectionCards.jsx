import InstrumentCard from "./InstrumentCard.jsx";

const InstrumentSelectionCards = ({ instruments, handleInstrumentSelect }) => {
  return (
    <div className="mt-16 md:mt-32 lg:mt-64 px-4 flex flex-col items-center mb-6">
      <h2 className="text-lg md:text-xl mb-6 text-lightgray text-center tracking-widest">
        Which instrument are you analyzing today?
      </h2>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6 justify-items-center w-full max-w-md lg:max-w-none">
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
