import mongoose from "mongoose";
import dotenv from "dotenv";
import connectDB from "../db.js";

dotenv.config();

const dropIndex = async () => {
  try {
    // Use your existing connectDB function to establish SSH tunnel + MongoDB connection
    await connectDB("development"); // or "production" if needed

    console.log("Attempting to drop sessionId_1 index...");

    const collection = mongoose.connection.db.collection("musavoicesessions");

    // Check existing indexes first
    const indexes = await collection.indexes();
    console.log("Current indexes:", indexes);

    // Drop the sessionId_1 index if it exists
    try {
      await collection.dropIndex("sessionId_1");
      console.log("✅ Successfully dropped sessionId_1 index");
    } catch (error) {
      if (error.code === 27) {
        console.log("ℹ️  Index 'sessionId_1' does not exist (already dropped)");
      } else {
        throw error;
      }
    }

    // Verify it's gone
    const updatedIndexes = await collection.indexes();
    console.log("Updated indexes:", updatedIndexes);

    await mongoose.connection.close();
    console.log("Connection closed");
    process.exit(0);
  } catch (error) {
    console.error("❌ Error:", error.message);
    if (mongoose.connection.readyState === 1) {
      await mongoose.connection.close();
    }
    process.exit(1);
  }
};

dropIndex();
