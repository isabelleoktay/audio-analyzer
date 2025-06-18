class TokenManager {
  constructor() {
    this.token = localStorage.getItem("audio_analyzer_token");
  }

  async getToken() {
    // If we don't have a token, generate one
    if (!this.token) {
      await this.generateToken();
    }
    return this.token;
  }

  async generateToken() {
    try {
      const response = await fetch(
        `${process.env.REACT_APP_PYTHON_SERVICE_BASE_URL}/python-service/auth/token`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to generate token");
      }

      const data = await response.json();
      this.token = data.token;
      localStorage.setItem("audio_analyzer_token", this.token);

      console.log("New JWT token generated");
      return this.token;
    } catch (error) {
      console.error("Error generating token:", error);
      throw error;
    }
  }

  async verifyToken() {
    if (!this.token) {
      return false;
    }

    try {
      const response = await fetch(
        `${process.env.REACT_APP_PYTHON_SERVICE_BASE_URL}/python-service/auth/verify`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ token: this.token }),
        }
      );

      const data = await response.json();
      return data.valid;
    } catch (error) {
      console.error("Error verifying token:", error);
      return false;
    }
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem("audio_analyzer_token");
    console.log("Token cleared");
  }

  async ensureValidToken() {
    const isValid = await this.verifyToken();
    if (!isValid) {
      console.log("Token invalid or expired, generating new one");
      await this.generateToken();
    }
    return this.token;
  }
}

export const tokenManager = new TokenManager();
