# SkyNote Audio Analyzer – Developer Guide

This guide provides a comprehensive overview of the SkyNote (formerly MuSA) web application, its architecture, and detailed instructions for developers to set up, run, extend, and deploy the project. It covers the three main components: front-end, back-end, and python-service.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack & Prerequisites](#tech-stack--prerequisites)
- [Local Development Setup](#local-development-setup)
- [Project Structure](#project-structure)
- [Data Storage](#data-storage)
- [Deployment Instructions](#deployment-instructions)
- [Testing](#testing)
- [Error Handling & Logs](#error-handling--logs)
- [Contributing & Code Standards](#contributing--code-standards)
- [Extending the Application](#extending-the-application)
- [Security Considerations](#security-considerations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Architecture Overview

This project is composed of three distinct services that communicate with each other to provide a full-stack experience. Understanding the data flow is key to working on the application.

**Data Flow Diagram:**

```
User (Browser)
   ⇅
Front-end (React)
   ⇅ HTTP API
Back-end (Node.js/Express)
   ⇅ HTTP API
Python Service (Flask)
   ⇅
MongoDB (Database)
```

**Component Description:**

- **Front-end:** A React application that provides the user interface. It communicates exclusively with the back-end via a REST API.
- **Back-end:** A Node.js/Express server that acts as the central hub. It serves the front-end's API requests, manages file uploads, and proxies analysis requests to the python-service. It is also responsible for all interactions with the MongoDB database.
- **Python Service:** A Flask microservice that handles all computationally intensive audio feature extraction. It exposes its own API that is consumed by the back-end.
- **MongoDB:** The database used for storing audio file metadata, extracted feature data, and user information.

---

## Tech Stack & Prerequisites

Ensure you have the following software installed on your machine. The specified versions are recommended to prevent "works on my machine" issues.

- **Node.js:** 18.x
- **npm:** 9.x or higher
- **Python:** 3.10 or higher
- **MongoDB:** 6.x or higher

You can check your versions with the following commands:

```bash
node -v
npm -v
python --version
# For MongoDB Shell 5.x+
mongosh --version
# For older mongo shell
mongo --version
```

---

## Local Development Setup

Follow these steps to get all three services running on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Configure Environment Variables

Each service (`front-end`, `back-end`, `python-service`) requires its own `.env` file in its root directory. Create these files and add the necessary variables.

**front-end/.env**

```env
REACT_APP_API_BASE_URL=http://localhost:8004/
```

**back-end/.env**

```env
PORT=8004
PYTHON_SERVICE_URL=http://localhost:8080
DEV_DB_URL=mongodb://<user>:<password>@127.0.0.1:27017/audio-analyzer-dev?authSource=admin
PROD_DB_URL=mongodb://<user>:<password>@127.0.0.1:27017/audio-analyzer-prod?authSource=admin

# Variables for remote DB connection via SSH tunnel (optional for local setup)
VPS_PRIVATE_KEY=/Users/<your-user>/.ssh/id_rsa
VPS_USERNAME=<your-vps-user>
```

**python-service/.env**

```env
FLASK_APP=app.py
FLASK_RUN_PORT=8080
```

### 3. Install Dependencies

Run `npm install` or `pip install` in each service's directory.

**Front-end:**

```bash
cd front-end
npm install
cd ..
```

**Back-end:**

```bash
cd back-end
npm install
cd ..
```

**Python Service:**

```bash
cd python-service
pip install -r requirements.txt
cd ..
```

### 4. Run the Application

Start each service in a separate terminal window. It's recommended to start them in this order:

**Start Python Service:**

```bash
cd python-service
flask run
```

**Start Back-end:**

```bash
cd back-end
npm start
```

**Start Front-end:**

```bash
cd front-end
npm start
```

### 5. Access the Application

Once all services are running, open your browser and navigate to `http://localhost:3000` (or the port your React app is running on).

---

## Project Structure

The repository is organized into three main directories, one for each service.

### python-service/

- **Purpose:** Performs all heavy audio analysis.
- **Key Files:** `app.py` (Flask API), `requirements.txt`, and modules inside `feature_extraction/`.

### back-end/

- **Purpose:** The core API that connects the front-end to the other services.
- **Key Files:** `index.js` (server entry), `controllers/`, `routes/`, and `models/` (Mongoose schemas).

### front-end/

- **Purpose:** The user-facing React application.
- **Key Files:** `src/App.jsx`, `src/components/`, `package.json`.

---

## Data Storage

The application uses MongoDB for storing audio file metadata and test subject information.

**What is stored:**

- Audio file metadata (filename, upload date, duration, user ID).
- Extracted feature data (pitch, dynamics, vibrato, etc.).
- User/Test subject information (demographics, experiment group).

**Schema Definitions:**

- Mongoose schemas defining the structure of the data are located in the `back-end/models/` directory (e.g., `Audio.js`, `TestSubject.js`).
- For a more detailed schema visualization, consider adding documentation to a `/docs` folder.

The backend connects to MongoDB using the connection string defined in its `.env` file. For secure remote connections in production, an SSH tunnel can be established using your VPS credentials.

---

## Deployment Instructions

While the setup above is for local development, production requires a more robust configuration.

- **Node.js Back-end:** Use a process manager like PM2 or systemd to ensure the Node.js service runs continuously and restarts on failure.
- **Python Service:** Use a production-grade WSGI server like Gunicorn behind a process manager like systemd or Supervisor.
- **Front-end:** Build the static React application (`npm run build`) and serve the generated files using a web server like Nginx.
- **Reverse Proxy:** Nginx is highly recommended to act as a reverse proxy. It can handle incoming HTTPS traffic, terminate SSL, and route requests to the appropriate backend service (e.g., requests to `/api/` go to the Node.js app, other requests serve the React app).

**Example Nginx Configuration:**

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    # SSL Certs
    ssl_certificate /path/to/your/fullchain.pem;
    ssl_certificate_key /path/to/your/privkey.pem;

    # Route API calls to the backend service
    location /api/ {
        proxy_pass http://localhost:8004/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Serve the static front-end files
    location / {
        root /var/www/skynote/front-end/build;
        try_files $uri $uri/ /index.html;
    }
}
```

**Environment Variable Differences:**

- Ensure you create separate `.env` files for production or use a secret management system.
- Use production database URLs, API keys, and other credentials.
- **NEVER** commit production secrets to version control.

## Error Handling & Logs

**Where logs go:**

- In production, Node.js and Python logs are typically managed by the process manager (e.g., `pm2 logs` or `journalctl -u your-service`).
- Nginx logs are usually found in `/var/log/nginx/`.
- In local development, all logs output to the console.

**Debugging Common Errors:**

- Python service not responding: Check the logs from Flask/Gunicorn. Ensure it's running and accessible from the backend server.
- MongoDB connection issues: Verify your connection string, IP whitelisting rules, and credentials in the `.env` file. If using an SSH tunnel, ensure it's active.
- CORS errors: Ensure the backend server has the correct CORS configuration to allow requests from the front-end domain.

## Extending the Application

To add a new audio feature or instrument, or update the available analysis buttons for each instrument, follow these steps:

### 1. Front-end (React)

- Edit `front-end/src/config/analysisButtons.js`:
  - Add or modify entries in `analysisButtonConfig` to specify which features (buttons) are available for each instrument.
  - Example:
    ```js
    export const analysisButtonConfig = {
       violin: [ ... ],
       voice: [ ... ],
       polyphonic: [ ... ],
       guitar: [
          { type: "left", label: "dynamics" },
          { type: "center", label: "pitch" },
          { type: "right", label: "tempo" },
       ],
    };
    ```
- The `AnalysisButtons.jsx` component will automatically render buttons for any instrument defined in this config.

### 2. Python Service (Flask)

- Implement the feature extraction logic for the new instrument/feature in `python-service/feature_extraction/`.
- Expose a new endpoint in `app.py` if needed, or update existing endpoints to handle the new instrument/feature.

### 3. End-to-End Connection

- When a user selects a feature button in the front-end, it triggers an API call to the backend, which then calls the python-service for analysis.
- The backend and python-service must both recognize the instrument and feature label for the request to succeed.

**Summary:**

- Update `analysisButtonConfig` in the front-end for UI changes.
- Update backend and python-service to support new instruments/features for full-stack integration.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
