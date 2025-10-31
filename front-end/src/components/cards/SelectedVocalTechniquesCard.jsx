const SelectedVocalTechniquesCard = ({ selectedTechniques }) => {
  if (!selectedTechniques.length) return null;

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 max-w-[200px] shadow-lg">
      <h2 className="text-xl font-bold text-lightpink mb-3">
        Selected Vocal Techniques
      </h2>
      <div className="flex flex-wrap gap-2">
        {selectedTechniques.map((tech, idx) => (
          <span
            key={idx}
            className="text-xs font-light text-gray-200 px-2 py-1 rounded"
          >
            {tech}
          </span>
        ))}
      </div>
    </div>
  );
};

export default SelectedVocalTechniquesCard;
