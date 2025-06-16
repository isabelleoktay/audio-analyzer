from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder
import os

# Global variables for SSH tunnel and MongoDB client
ssh_tunnel = None
client = None
db = None

def start_ssh_tunnel():
    """Starts the SSH tunnel if a private key is provided."""
    global ssh_tunnel
    VPS_PRIVATE_KEY = os.getenv("VPS_PRIVATE_KEY")
    VPS_USERNAME = os.getenv("VPS_USERNAME")
    VPS_HOST = os.getenv("VPS_HOST", "appskynote.com")
    VPS_PORT = int(os.getenv("VPS_PORT", 22))

    if VPS_PRIVATE_KEY:
        try:
            ssh_tunnel = SSHTunnelForwarder(
                (VPS_HOST, VPS_PORT),
                ssh_username=VPS_USERNAME,
                ssh_private_key=os.path.expanduser(VPS_PRIVATE_KEY),
                remote_bind_address=("127.0.0.1", 27017),
                local_bind_address=("127.0.0.1", 27017),
            )
            ssh_tunnel.start()
            print("SSH tunnel established.")
        except Exception as e:
            print(f"Failed to establish SSH tunnel: {e}")
            raise
    else:
        print("Private key not found. Connecting directly to MongoDB.")

def connect_to_mongo():
    """Connects to MongoDB and returns the database object."""
    global client, db
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    try:
        client = MongoClient(MONGO_URI)
        db = client["audioAnalyzerDB"] 
        print("MongoDB connected successfully!")
        return db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

def close_ssh_tunnel():
    """Closes the SSH tunnel if it is active."""
    global ssh_tunnel
    if ssh_tunnel:
        ssh_tunnel.stop()
        print("SSH tunnel closed.")