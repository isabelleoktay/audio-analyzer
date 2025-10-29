import { useState } from "react";

const SimilarityScoreCard = ({
  similarityScore = 75,
  bestScores = [92, 88, 85, 83, 80],
}) => {
  const [expanded, setExpanded] = useState(false);
  const topScore = Math.max(...bestScores);

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className={`
        bg-white/10 backdrop-blur-md rounded-2xl p-5 shadow-lg
        w-full max-w-[200px] transition-all duration-500 ease-in-out
        cursor-pointer select-none
        hover:shadow-lightpink/40 hover:scale-[1.02]
      `}
    >
      {/* Title */}
      <h2 className="text-xl font-bold text-lightpink mb-2">
        Similarity Score
      </h2>

      {/* Current score */}
      <div className="text-4xl font-extrabold text-white mb-1">
        {similarityScore !== null ? `${similarityScore.toFixed(2)}%` : "N/A"}
      </div>

      {/* Best score */}
      <div className="text-sm text-gray-300">Best: {topScore.toFixed(2)}%</div>

      {/* Expandable section */}
      <div
        className={`transition-all duration-500 ease-in-out overflow-hidden ${
          expanded ? "max-h-48 opacity-100 mt-3" : "max-h-0 opacity-0"
        }`}
      >
        <h3 className="text-sm font-semibold text-lightpink mb-2">
          Your Top 5 Scores
        </h3>
        <ul className="text-white space-y-1 text-sm">
          {bestScores.map((score, index) => (
            <li key={index}>
              {index + 1}. {score.toFixed(2)}%
            </li>
          ))}
        </ul>
      </div>

      {/* Hint */}
      <div className="text-xs text-gray-400 mt-2 italic text-center">
        {expanded ? "Click to collapse" : "Click to view top 5 scores"}
      </div>
    </div>
  );
};

export default SimilarityScoreCard;
