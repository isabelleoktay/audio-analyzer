import { useState, useEffect } from "react";

const useLocalStorageState = (key, initialValue, persist = true) => {
  const [value, setValue] = useState(() => {
    if (persist) {
      const stored = localStorage.getItem(key);
      if (stored !== null) {
        try {
          return JSON.parse(stored);
        } catch (error) {
          return stored;
        }
      }
    }
    return initialValue;
  });

  useEffect(() => {
    if (persist) {
      localStorage.setItem(key, JSON.stringify(value));
    }
  }, [key, value, persist]);

  useEffect(() => {
    if (persist) {
      const onStorageChange = (e) => {
        if (e.key === key) {
          try {
            setValue(JSON.parse(e.newValue));
          } catch (error) {
            setValue(e.newValue);
          }
        }
      };
      window.addEventListener("storage", onStorageChange);
      return () => window.removeEventListener("storage", onStorageChange);
    }
  }, [key, persist]);

  return [value, setValue];
};

export default useLocalStorageState;
