# SkyNote Audio Analyzer

## Setting Up the Local Application

### 1. Environment Variables

There are three directories that require environment variables:

1. **Frontend Directory:** You should put this .env in the root of the frontend directory. The contents of this file in my configuration look like the following example. Your .env file should contain the same information (unless you change the ports of the Python and Nodejs APIs accordingly):

   ```
   REACT_APP_API_BASE_URL=http://localhost:8004/
   REACT_APP_PYTHON_SERVICE_BASE_URL=http://localhost:8080/
   ```

2. **Backend Directory:** You should put this .env file in the root of the backend directory. The contents of this file in my configuration look like the following example. Your file should not contain the exact same information, as you should fill out the MongoDB URI with your own username and password as well as the VPS private key and username with your own key and name. **NOTE:** MongoDB is not being used in the application yet, so there is no need to actually use the MongoDB URI. The port information should stay the same unless you've changed the API ports accordingly.

   ```
   PORT=8004
   PYTHON_SERVICE_URL=http://localhost:8080
   MONGODB_URI=mongodb://<username>:<password>@127.0.0.1:27017/audioAnalyzerDB?serverSelectionTimeoutMS=5000&connectTimeoutMS=10000&authSource=skynote&authMechanism=SCRAM-SHA-256
   VPS_PRIVATE_KEY = /Users/<yourname>/.ssh/id_rsa
   VPS_USERNAME = <yourname>
   ```

3. **Python Service Directory:** You should put this .env file in the root of the python service directory. The contents of this .env file should look like the following:

   ```
   FLASK_APP=app.py
   ```

### 2. Running the application

The commands to run the application are located in `package.json`. You can use `npm start` to run the application. However, you will need to ensure that all dependencies are installed. Since this application used npm and python, you will need to have the node package manager and a python environment installed.

1. **Node dependencies:** You will need to install the node dependencies on both the frontend and backend directories of the application.

   - **To install the frontend dependencies**, `cd` to the root of the frontend directory and use the command `npm i`. This should install the node dependencies. If `npm i` doesn't work, you may need to enable legacy dependencies with `npm i --legacy-peer-deps`.
   - **To install the backend dependencies**: `cd` to the root of the backend directory and use the command `npm i`. This should install the node dependencies. If `npm i` doesn't work, you may need to enable legacy dependencies with `npm i --legacy-peer-deps`.

2. **Python dependencies**: You will need to have a python environment activated to run this application locally. Once this environment is activated, you should `cd` to the python service directory and run the following command to install the dependencies:

   ```
   pip install -r requirements.txt
   ```

Once again, when you have installed all the necessary dependencies, you may `cd` to the root directory of this project and run the command `npm start` to run the application locally.

## Deploying and Updating the Application

### 1. Make Code Changes

- Modify the code in the backend (Node.js, Python, etc.) or frontend as required. After making changes, push your code to GitHub.

### 2. SSH into the Server

- Connect to your server via SSH:

```
ssh username@appskynote.com
```

### 3. Pull Latest Changes

```
cd /home/isabelle/audio-analyzer
git pull origin main
```

### 4. Install and Set Up Python Service

- **If you have already set up the Python service, you can ignore this step and move on to Step 5.** You can check if the Python service is running by using the following command:

```
sudo supervisorctl status python-service
```

- Python service is running if you receive a response like:

```
python-service                   RUNNING   pid 3590747, uptime 0:40:31
```

#### 4.1 Setting up the Python virtual environment

- To install the Python virtual environment, you can run the following command from the `audio-analyzer` root directory:

```
npm run create-venv
```

- This will create the Python virtual environment and install all dependencies in `requirements.txt`.
- Once the virtual environment is created, you should activate the python virtual environment.

#### 4.2 Preparing the application for production deployment

- This application uses Gunicorn, a WSGI HTTP server to serve Python applications in a production environment. To activate the Python virtual environment and install Gunicorn, you should run the following code:

```
source /home/isabelle/audio-analyzer/python-service/venv/bin/activate
pip install gunicorn
```

- The python service is managed with **Supervisor**. Ensure that **Supervisor** is installed and running. You can find the python servisor Supervisor configuration file in `/etc/supervisor/conf.d/python-service.conf`. This file contains the paths to `std_out` and `stderr` files. The current configuration of the Supervisor configuration file contains the following:

```
[program:python-service]
directory=/home/isabelle/audio-analyzer/python-service
command=/home/isabelle/audio-analyzer/python-service/venv/bin/gunicorn --bind 0.0.0.0:8080 app:app
autostart=true
autorestart=true
stderr_logfile=/var/log/python-service.err.log
stdout_logfile=/var/log/python-service.out.log
```

- Once you have the configuration in place (or if you ever change the configuration), you need to run the following code:

```
# Reread Supervisor configurations to include the new python-service config
sudo supervisorctl reread

# Update Supervisor to apply the new config
sudo supervisorctl update

# Start the Python service
sudo supervisorctl start python-service
```

### 5. Update and Monitor Python Service

- If you make changes to the Python code and need to restart the service, you can restart it with the following command:

```
sudo supervisorctl restart python-service
```

- You can monitor the logs to check if the service is running correctly or if any issues arise. The logs are located in the /var/log directory:

```
# Check status
sudo supervisorctl status python-service

# View standard output logs
tail -f  /var/log/python-service.out.log

# View error logs
tail -f  /var/log/python-service.err.log
```

### 6. Install and Set Up Node.js Back End

- **If you have already set up the Node.js back end service, you can ignore this step and move on to Step 7.**
- Supervisor manages the Node.js backend for the audio analyzer and ensures it starts automatically, restarts if it crashes, and keeps logs for debugging.
- The configuration file for Supervisor is located in `/etc/supervisor/conf.d/audio-analyzer-backend.conf` and should look like this:

```
[program:audio-analyzer-backend]
command=/usr/bin/node /home/isabelle/audio-analyzer/back-end/index.js
directory=/home/isabelle/audio-analyzer/back-end
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/audio-analyzer-backend-stderr.log
stdout_logfile=/var/log/supervisor/audio-analyzer-backend-stdout.log
environment=NODE_ENV="production",PORT=8004
user=isabelle
```

- Before running the Node.js service, make sure you install the necessary dependencies:

```
cd /home/isabelle/audio-analyzer/back-end
npm install
```

- After setting up the Supervisor configuration, reload Supervisor to apply the new configuration and start the Node.js service:

```
# Reread Supervisor configurations to include the new nodejs-backend config
sudo supervisorctl reread

# Update Supervisor to apply the new config
sudo supervisorctl update

# Start the Node.js backend service
sudo supervisorctl start audio-analyzer-backend
```

### 7. Update and Monitor Node.js Back End

- If you make changes to the Node.js back end code and need to restart the service, you can restart it with the following command:

```
sudo supervisorctl restart audio-analyzer-backend
```

- You can monitor the logs to check if the service is running correctly or if any issues arise. The logs are located in the /var/log directory:

```
# Check status
sudo supervisorctl status audio-analyzer-backend

# View standard output logs
tail -f  /var/log/supervisor/audio-analyzer-backend-stdout.log

# View error logs
tail -f  /var/log/supervisor/audio-analyzer-backend-stderr.log
```

### 8. Set Up and Update the React Front End with Nginx

- The React front end build files are located in `/etc/nginx/sites-available/audio-analyzer.frontend`.
- Every time you update the front end code, you first need to pull in the changes to the server copy:

```
cd /home/isabelle/audio-analyzer
git pull origin main
```

- Next, you need to build the static files that will be hosted by nginx:

```
cd /home/isabelle/audio-analyzer/front-end
npm run build
```

- From this directory, you must copy the build over to `/etc/nginx/sites-available/audio-analyzer.frontend`, where the files are located for nginx:

```
sudo cp -r build/* /var/www/html/audio-analyzer.frontend/
```

- The nginx configuration file is located in `/etc/nginx/sites-available/default`. You can access its contents with the following command:

```
sudo nano /etc/nginx/sites-available/default
```

- If you haven't already done so, ensure that there is a symbolic link in the `sites-enabled` directory to enable this site:

```
sudo ln -s /etc/nginx/sites-available/audio-analyzer.frontend /etc/nginx/sites-enabled/
```

- Whenever you update the nginx configuration file, you must test the configuration to make sure everything is set up correctly:

```
sudo nginx -t
```

- If the test passes, reload Nginx to apply the changes (either from the nginx configuration file or from code changes in the front-end):

```
sudo systemctl reload nginx
```

- You can monitor Nginx logs using the following code:

```
# Access logs (requests to your frontend)
tail -f /var/log/nginx/access.log

# Error logs
tail -f /var/log/nginx/error.log
```
