// components/InstrumentCard.jsx
const InstrumentCard = ({ Icon, label, onClick }) => (
  <button
    onClick={onClick}
    className="bg-lightgray/25 hover:bg-gradient-to-b hover:from-darkpink/25 hover:to-electricblue/25 rounded-3xl p-6 text-lightgray flex flex-col items-center aspect-square"
  >
    <Icon className="w-40 h-40 fill-current" />
    <span className="text-sm tracking-widest">{label}</span>
  </button>
);

export default InstrumentCard;
