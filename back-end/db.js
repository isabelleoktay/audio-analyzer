import { createTunnel } from "tunnel-ssh";
import mongoose from "mongoose";
import fs from "fs";
import dotenv from "dotenv";
dotenv.config();

const private_key = process.env.VPS_PRIVATE_KEY;

const connectDB = async (url) => {
  mongoose.set("strictQuery", false);

  if (private_key) {
    const privateKey = fs.readFileSync(private_key);

    const sshOptions = {
      host: "appskynote.com",
      port: 22,
      username: process.env.VPS_USERNAME,
      privateKey: privateKey,
    };

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
    await createSSHTunnel(sshOptions, 27017);
    console.log("SSH tunnel established.");
  } else {
    console.log("Private key not found.");
  }

  try {
    await mongoose.connect(url);
    console.log("MongoDB connected successfully!");
    console.log(mongoose.connection.readyState);
  } catch (err) {
    console.log("MongoDB connection failed:", err);
  }
};

export default connectDB;
