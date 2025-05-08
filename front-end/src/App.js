// import AudioAnalyzer from "./pages/AudioAnalyzer";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import NavBar from "./components/NavBar";
import Analyzer from "./pages/Analyzer";

const App = () => {
  return (
    <div className="min-h-screen bg-radial from-bluegray to-blueblack">
      <Router>
        <Layout>
          <NavBar />
          <Routes>
            <Route path="/" element={<Analyzer />} />
            <Route path="*" element={<div>404 Not Found</div>} />
          </Routes>
        </Layout>
      </Router>
    </div>
  );
};

export default App;
