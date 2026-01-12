// src/components/Layout.jsx
import { useLocation } from "react-router-dom";
import NavBar from "./NavBar.jsx";

const Layout = ({
  children,
  handleReset,
  uploadsEnabled,
  setUploadsEnabled,
  setTooltipMode,
  tooltipMode,
}) => {
  const { pathname } = useLocation();
  const hideNav = pathname.startsWith("/musavoice-testing");

  return (
    <div className="px-6 md:px-12 lg:px-24 xl:px-32 max-w-screen-2xl mx-auto w-full flex-grow pb-20 min-h-screen">
      {!hideNav && (
        <NavBar
          handleReset={handleReset}
          uploadsEnabled={uploadsEnabled}
          setUploadsEnabled={setUploadsEnabled}
          setTooltipMode={setTooltipMode}
          tooltipMode={tooltipMode}
        />
      )}
      {children}
    </div>
  );
};

export default Layout;
