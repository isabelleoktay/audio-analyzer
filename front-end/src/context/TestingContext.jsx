import { createContext, useReducer } from "react";
import { testingReducer, initialState } from "../hooks/useTestingReducer";

export const TestingContext = createContext();

export const TestingProvider = ({ children }) => {
  const [state, dispatch] = useReducer(testingReducer, initialState);

  return (
    <TestingContext.Provider value={{ state, dispatch }}>
      {children}
    </TestingContext.Provider>
  );
};
