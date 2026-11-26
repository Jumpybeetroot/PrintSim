import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { installPagesApiBridge } from "./utils/pagesApiBridge";
import "@fontsource-variable/inter";
import "./styles.css";

installPagesApiBridge();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
