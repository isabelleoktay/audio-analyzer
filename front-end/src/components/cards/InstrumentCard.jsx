// components/InstrumentCard.jsx
const InstrumentCard = ({ Icon, label, onSelect }) => (
  <button
    onClick={onSelect}
    className="bg-lightgray/25 hover:bg-gradient-to-b hover:from-darkpink/25 hover:to-electricblue/25 rounded-3xl p-8 text-lightgray flex flex-col items-center aspect-square w-fit"
  >
    <svg
      className="w-32 h-32 lg:w-52 lg:h-52 mb-4"
      viewBox="0 0 800 800"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="icon-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#FF89BB" />
          <stop offset="100%" stopColor="#90F1EF" />
        </linearGradient>
      </defs>
      <Icon className="w-full h-full fill-[url(#icon-gradient)]" />
    </svg>
    <span className="tracking-widest">{label}</span>
  </button>
);

export default InstrumentCard;
