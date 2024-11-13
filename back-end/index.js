import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import bodyParser from "body-parser";
import "express-async-errors";
import morgan from "morgan";
import { dirname } from "path";
import { fileURLToPath } from "url";
import path from "path";
import processAudioRouter from "./routes/processAudioRoutes.js";

dotenv.config();

const app = express();

// Enable CORS for all routes
app.use(cors());

// Middleware
app.use(express.json({ limit: "16mb" })); // For JSON payloads
app.use(express.urlencoded({ limit: "16mb", extended: true })); // For URL-encoded data
app.use(bodyParser.urlencoded({ extended: false })); // Parse application/x-www-form-urlencoded
app.use(bodyParser.json()); // Parse application/json

if (process.env.NODE_ENV !== "production") {
  app.use(morgan("dev")); // Log HTTP requests in the console
}

// set up express to serve static files from React front end
// const __dirname = dirname(fileURLToPath(import.meta.url));
// app.use(express.static(path.resolve(__dirname, "../front-end/build")));

app.get("/api/", (_, res) => {
  res.send({ msg: "Welcome to the Audio Analyzer API!" });
});

app.use("/api", processAudioRouter);

const port = process.env.PORT || 5000;

const start = async () => {
  try {
    app.listen(port, () =>
      console.log(`Server is listening on port ${port}...`)
    );
  } catch (error) {
    console.log("Error starting server:", error);
  }
};

start();
