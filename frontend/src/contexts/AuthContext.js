import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext({});

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState(null);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  const API = `${BACKEND_URL}/api`;

  useEffect(() => {
    // Check for existing authentication on mount
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      // Check URL fragment for session_id (Google OAuth)
      const fragment = window.location.hash.substring(1);
      const params = new URLSearchParams(fragment);
      const sessionId = params.get('session_id');

      if (sessionId) {
        // Process Google OAuth session
        await processGoogleSession(sessionId);
        return;
      }

      // Check for existing session token in cookies or localStorage
      const existingToken = getCookie('session_token') || localStorage.getItem('auth_token');
      if (existingToken) {
        await validateToken(existingToken);
      }
    } catch (error) {
      console.error('Auth check failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const processGoogleSession = async (sessionId) => {
    try {
      setLoading(true);
      console.log('Processing Google OAuth session...');

      const response = await axios.get(`https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data`, {
        headers: {
          'X-Session-ID': sessionId
        }
      });

      const { session_token, ...userData } = response.data;

      // Store session token
      setCookie('session_token', session_token, 7); // 7 days
      setToken(session_token);

      // Store user in backend database
      await storeUserInBackend(userData, session_token);

      // Clean URL
      window.history.replaceState({}, document.title, window.location.pathname);

      console.log('Google OAuth authentication successful');
    } catch (error) {
      console.error('Google OAuth session processing failed:', error);
      throw error;
    }
  };

  const storeUserInBackend = async (userData, sessionToken) => {
    try {
      // Call our backend to store the user session
      const response = await axios.post(`${API}/auth/google/session`, {
        user_data: userData,
        session_token: sessionToken
      });

      setUser(response.data.user);
      setToken(sessionToken);
    } catch (error) {
      console.error('Failed to store user in backend:', error);
      throw error;
    }
  };

  const validateToken = async (authToken) => {
    try {
      const response = await axios.get(`${API}/auth/me`, {
        headers: {
          Authorization: `Bearer ${authToken}`
        }
      });

      setUser(response.data);
      setToken(authToken);
      return true;
    } catch (error) {
      console.error('Token validation failed:', error);
      // Clear invalid token
      localStorage.removeItem('auth_token');
      deleteCookie('session_token');
      return false;
    }
  };

  const login = async (email, password) => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/auth/login`, {
        email,
        password
      });

      const { access_token, user: userData } = response.data;
      
      // Store token and user data
      localStorage.setItem('auth_token', access_token);
      setToken(access_token);
      setUser(userData);

      return { success: true, user: userData };
    } catch (error) {
      const message = error.response?.data?.detail || 'Login failed';
      return { success: false, error: message };
    } finally {
      setLoading(false);
    }
  };

  const register = async (email, password, fullName, referralCode = '') => {
    try {
      setLoading(true);
      const response = await axios.post(`${API}/auth/register`, {
        email,
        password,
        full_name: fullName,
        referral_code: referralCode || undefined
      });

      const { access_token, user: userData } = response.data;
      
      // Store token and user data
      localStorage.setItem('auth_token', access_token);
      setToken(access_token);
      setUser(userData);

      return { success: true, user: userData };
    } catch (error) {
      const message = error.response?.data?.detail || 'Registration failed';
      return { success: false, error: message };
    } finally {
      setLoading(false);
    }
  };

  const loginWithGoogle = () => {
    // Redirect to Emergent Google OAuth
    const redirectUrl = `${window.location.origin}/dashboard`;
    const authUrl = `https://auth.emergentagent.com/?redirect=${encodeURIComponent(redirectUrl)}`;
    window.location.href = authUrl;
  };

  const logout = async () => {
    try {
      // Call backend logout if we have a session token
      if (getCookie('session_token')) {
        await axios.post(`${API}/auth/logout`, {}, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear all auth data
      localStorage.removeItem('auth_token');
      deleteCookie('session_token');
      setToken(null);
      setUser(null);
    }
  };

  const refreshUserData = async () => {
    if (token) {
      try {
        const response = await axios.get(`${API}/auth/me`, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        setUser(response.data);
      } catch (error) {
        console.error('Failed to refresh user data:', error);
      }
    }
  };

  // Cookie utilities
  const setCookie = (name, value, days) => {
    const expires = new Date();
    expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
    document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/;secure;samesite=none`;
  };

  const getCookie = (name) => {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for (let i = 0; i < ca.length; i++) {
      let c = ca[i];
      while (c.charAt(0) === ' ') c = c.substring(1, c.length);
      if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
    }
    return null;
  };

  const deleteCookie = (name) => {
    document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/;secure;samesite=none`;
  };

  const value = {
    user,
    token,
    loading,
    login,
    register,
    logout,
    loginWithGoogle,
    refreshUserData,
    isAuthenticated: !!user
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};