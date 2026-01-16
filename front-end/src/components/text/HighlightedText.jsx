const escapeRegex = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const normalizeHighlights = (highlightWords = []) =>
  highlightWords.map((h) =>
    typeof h === "string"
      ? { phrase: h, occurrences: null }
      : { phrase: h.phrase ?? "", occurrences: h.occurrences ?? null }
  );

const HighlightedText = ({
  text = "",
  highlightWords = [],
  highlightClass = "bg-electricblue/70 text-blueblack",
  defaultClass = "bg-darkpink/70 text-blueblack",
  highlightLabel = "vibrato",
  highlightLabelColor = "text-electricblue",
  defaultLabel = "straight",
  defaultLabelColor = "text-darkpink",
  className = "",
}) => {
  if (!text || highlightWords.length === 0)
    return <p className={defaultClass}>{text}</p>;

  const parts = [];
  const highlights = [];
  const normalized = normalizeHighlights(highlightWords);

  normalized.forEach(({ phrase, occurrences }) => {
    if (!phrase) return;
    const regex = new RegExp(escapeRegex(phrase), "gi");
    let match;
    let occurrence = 0;
    while ((match = regex.exec(text)) !== null) {
      occurrence += 1;
      const shouldHighlight = !occurrences || occurrences.includes(occurrence);
      if (!shouldHighlight) continue;
      highlights.push({
        start: match.index,
        end: match.index + match[0].length,
      });
    }
  });

  highlights.sort((a, b) => a.start - b.start);
  let lastIndex = 0;
  highlights.forEach((h) => {
    if (lastIndex < h.start) {
      parts.push({ type: "text", content: text.slice(lastIndex, h.start) });
    }
    parts.push({ type: "highlight", content: text.slice(h.start, h.end) });
    lastIndex = h.end;
  });
  if (lastIndex < text.length) {
    parts.push({ type: "text", content: text.slice(lastIndex) });
  }

  let highlightLabelInserted = false;
  let defaultLabelInserted = false;

  return (
    <p className={`inline-block ${className}`}>
      {parts.map((part, i) => {
        if (part.type === "highlight") {
          const needsLabel = highlightLabel && !highlightLabelInserted;
          highlightLabelInserted = true;

          return (
            <span key={`h-${i}`} className="inline-block">
              {needsLabel && (
                <div
                  className={`text-center font-semibold text-xs mb-1 ${highlightLabelColor}`}
                >
                  {highlightLabel}
                </div>
              )}
              <span className={`${highlightClass} p-0.5 rounded-sm mr-1`}>
                {part.content}
              </span>
            </span>
          );
        }

        const needsLabel = defaultLabel && !defaultLabelInserted;
        defaultLabelInserted = true;

        return (
          <span key={`t-${i}`} className="inline-block">
            {needsLabel && (
              <div
                className={`text-center font-semibold text-xs mb-1 ${defaultLabelColor} mx-2`}
              >
                {defaultLabel}
              </div>
            )}
            <span className={`${defaultClass} p-0.5 rounded-sm mr-1`}>
              {part.content}
            </span>
          </span>
        );
      })}
    </p>
  );
};

export default HighlightedText;
