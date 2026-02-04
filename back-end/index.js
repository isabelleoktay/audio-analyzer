import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import bodyParser from "body-parser";
import "express-async-errors";
import morgan from "morgan";

import uploadAudioRouter from "./routes/uploadAudioRoutes.js";
import uploadTestSubjectRouter from "./routes/uploadTestSubjectRoutes.js";
import uploadFeedbackRouter from "./routes/uploadFeedbackRoutes.js";
import uploadMusaVoiceSessionRouter from "./routes/uploadMusaVoiceSessionRoutes.js";
import musaUserStudyRouter from "./routes/uploadMusaUserStudyRoutes.js";

import connectDB from "./db.js";

dotenv.config();

// Initialize the Express application
const app = express();

// Enable Cross-Origin Resource Sharing (CORS) for all routes
app.use(cors());

// Middleware to parse incoming request bodies
app.use(express.json({ limit: "16mb" }));
app.use(express.urlencoded({ limit: "16mb", extended: true }));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

const env = process.env.NODE_ENV || "development";
// Enable HTTP request logging in non-production environments
if (env !== "production") {
  app.use(morgan("dev"));
}

// Define a base API route for health checks or basic information
app.get("/api/", (_, res) => {
  res.send({ msg: "Welcome to the Audio Analyzer API!" });
});

// Register routers for handling specific API routes
app.use("/api", uploadAudioRouter);
app.use("/api", uploadTestSubjectRouter);
app.use("/api", uploadFeedbackRouter);
app.use("/api", uploadMusaVoiceSessionRouter);
app.use("/api", musaUserStudyRouter);

const port = process.env.PORT || 5000;

/**
 * Starts the server and connects to the MongoDB database.
 */
const start = async () => {
  try {
    await connectDB(env);
    app.listen(port, () =>
      console.log(`Server is listening on port ${port}...`)
    );
  } catch (error) {
    console.log("Error starting server:", error);
  }
};

start();
