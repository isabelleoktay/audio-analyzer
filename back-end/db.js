import { createTunnel } from "tunnel-ssh";
import mongoose from "mongoose";
import fs from "fs";
import dotenv from "dotenv";
dotenv.config();

const private_key = process.env.VPS_PRIVATE_KEY;

/**
 * Connects to the Audio Analyzer MongoDB database.
 * If a private key is provided, it establishes an SSH tunnel to securely connect to the database.
 *
 * @param {string} url - The MongoDB connection string.
 */
const connectDB = async (env) => {
  mongoose.set("strictQuery", false);

  const url =
    env === "production" ? process.env.PROD_DB_URL : process.env.DEV_DB_URL; // Use different URLs for production and development

  if (private_key) {
    const privateKey = fs.readFileSync(private_key);

    const sshOptions = {
      host: "appskynote.com",
      port: 22,
      username: process.env.VPS_USERNAME,
      privateKey: privateKey,
    };

    /**
     * Creates an SSH tunnel to forward traffic to the MongoDB server.
     *
     * @param {object} sshOptions - SSH connection options.
     * @param {number} port - The port to forward traffic to (default MongoDB port is 27017).
     * @param {boolean} [autoClose=true] - Whether the tunnel should automatically close when the process exits.
     * @returns {Promise} - Resolves when the tunnel is successfully created.
     */
    const createSSHTunnel = (sshOptions, port, autoClose = true) => {
      let forwardOptions = {
        srcAddr: "127.0.0.1",
        srcPort: port,
        dstAddr: "127.0.0.1",
        dstPort: port,
      };

      let tunnelOptions = {
        autoClose: autoClose,
      };

      let serverOptions = {
        host: "127.0.0.1",
        port: port,
      };

      return createTunnel(
        tunnelOptions,
        serverOptions,
        sshOptions,
        forwardOptions
      );
    };

    // Establish the SSH tunnel
    await createSSHTunnel(sshOptions, 27017);
    console.log("SSH tunnel established.");
  } else {
    console.log("Private key not found.");
  }

  // Attempt to connect to MongoDB
  try {
    await mongoose.connect(url);
    console.log("MongoDB connected successfully!");
    console.log("Connected to database:", mongoose.connection.name);
    console.log(mongoose.connection.readyState);
  } catch (err) {
    console.log("MongoDB connection failed:", err);
  }
};

export default connectDB;
