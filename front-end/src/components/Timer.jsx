import { useState, useEffect } from "react";

const Timer = ({ initialTime = 300, onTimerFinish }) => {
  const [timeLeft, setTimeLeft] = useState(initialTime);

  useEffect(() => {
    // Start the countdown when the component is mounted
    const timerInterval = setInterval(() => {
      setTimeLeft((prevTime) => {
        if (prevTime <= 0) {
          clearInterval(timerInterval);
          if (onTimerFinish) {
            // Defer the state update in the parent component
            setTimeout(() => {
              onTimerFinish();
            }, 0);
          }
          return 0;
        }
        return prevTime - 1;
      });
    }, 1000);

    return () => clearInterval(timerInterval); // Cleanup on unmount
  }, [onTimerFinish]);

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, "0")}:${remainingSeconds
      .toString()
      .padStart(2, "0")}`;
  };

  return (
    <div className="bg-blueblack/50 text-sm text-warmyellow rounded-3xl px-4 py-2 w-fit">
      {formatTime(timeLeft)}
    </div>
  );
};

export default Timer;
