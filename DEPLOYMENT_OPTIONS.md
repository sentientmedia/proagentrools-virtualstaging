# ProAgentTools Deployment & Hosting Guide

**Author:** Manus AI  
**Date:** November 23, 2025

## 1. Introduction: Understanding Your Application's Architecture

This document provides a detailed guide for deploying your **ProAgentTools** application to a development or production environment. It is essential to first understand that your application is not a single, monolithic entity. It consists of three distinct components, each with its own specific hosting requirements:

1.  **Frontend:** A **React** single-page application (SPA) that runs in the user's browser. It needs to be hosted on a service optimized for serving static files quickly and globally.

2.  **Backend:** A **Python (FastAPI)** server that handles business logic, user authentication, database interactions, and communication with the Stripe and Replicate APIs.

3.  **Database:** A **MongoDB** database that stores all your data, including user information, credits, transactions, and referrals.

Because of this modern, decoupled architecture, a traditional shared hosting provider like Siteground is not the ideal choice. Such platforms are typically optimized for PHP-based applications like WordPress and lack the specific environment and tooling needed to run a multi-part application like yours efficiently. While it might be possible on a high-end VPS plan with manual configuration, it is far more complex and less cost-effective than using dedicated services for each component.

This guide will explore the recommended modern approach and also discuss how you might (or might not) use Siteground and your Supabase account.

---

## 2. Recommended Approach: The Modern, Best-Practice Stack

For the best performance, developer experience, scalability, and cost-effectiveness (leveraging generous free tiers), we strongly recommend using a combination of specialized cloud services. This is the standard for modern web development.

| Component | Recommended Service | Why It's a Good Fit | Free Tier Availability |
| :--- | :--- | :--- | :--- |
| **Frontend (React)** | **Vercel** | Optimized for Next.js/React, global CDN, automatic CI/CD from Git, preview deployments. | **Excellent:** Free for personal projects and small teams. [1] |
| **Backend (FastAPI)** | **Render** | Natively supports Python/FastAPI, simple Git-based deploys, managed databases, private networking. | **Good:** Free tier for web services (with spin-down). [2] |
| **Database (MongoDB)** | **MongoDB Atlas** | Fully managed MongoDB hosting, automated backups, scaling, and a perpetual free M0 cluster. | **Excellent:** Perpetual free tier perfect for development. [3] |

### Step-by-Step Deployment Guide (Recommended Stack)

#### **Part A: Deploying the Database on MongoDB Atlas**

1.  **Create an Account:** Sign up for a free account at [cloud.mongodb.com](https://cloud.mongodb.com).
2.  **Create a Free Cluster:** Follow the prompts to create a new **M0 Sandbox** cluster. This is free forever. You can choose any cloud provider and region.
3.  **Configure Security:**
    *   **Database User:** Create a database user with a secure password. You will need these credentials for your backend's connection string.
    *   **Network Access:** For a development environment, the simplest approach is to allow access from anywhere. Go to `Network Access` and click `Add IP Address`, then select `Allow Access from Anywhere` (0.0.0.0/0). For production, you should restrict this to your backend server's IP address.
4.  **Get Connection String:** Go to your cluster's `Overview` page, click `Connect`, select `Drivers`, and copy the **Connection String**. It will look like this:
    `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`
    *   Replace `<username>` and `<password>` with the credentials you created. This full string will be your `MONGO_URL` environment variable.

#### **Part B: Deploying the Backend on Render**

1.  **Create an Account:** Sign up at [render.com](https://render.com) using your GitHub account.
2.  **Create a New Web Service:** From the dashboard, click `New +` > `Web Service`.
3.  **Connect Your Repository:** Grant Render access and select your `proagentrools-virtualstaging` repository.
4.  **Configure the Service:**
    *   **Name:** Give your service a name (e.g., `proagent-backend`).
    *   **Root Directory:** Set this to `backend`.
    *   **Runtime:** Render will auto-detect Python.
    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
    *   **Instance Type:** Choose the `Free` tier.
5.  **Add Environment Variables:** Go to the `Environment` tab and add all the keys from your `.env.example` file, including `MONGO_URL` (from MongoDB Atlas), your Stripe keys, and your Replicate API token. **Crucially, set `FRONTEND_URL` to the URL you will get from Vercel in the next step.**
6.  **Deploy:** Click `Create Web Service`. Render will pull your code, build it, and deploy it. You will get a public URL (e.g., `https://proagent-backend.onrender.com`).

#### **Part C: Deploying the Frontend on Vercel**

1.  **Create an Account:** Sign up at [vercel.com](https://vercel.com) using your GitHub account.
2.  **Create a New Project:** From your dashboard, click `Add New...` > `Project`.
3.  **Connect Your Repository:** Import your `proagentrools-virtualstaging` repository.
4.  **Configure the Project:**
    *   **Framework Preset:** Vercel will auto-detect `Create React App`.
    *   **Root Directory:** Set this to `frontend`.
    *   Leave the build and output settings as their defaults.
5.  **Add Environment Variables:** Expand the `Environment Variables` section and add `REACT_APP_BACKEND_URL` and `REACT_APP_API_URL`. The value for both should be the backend URL you got from Render (e.g., `https://proagent-backend.onrender.com`).
6.  **Deploy:** Click `Deploy`. Vercel will build and deploy your site to its global CDN. You will get a public URL (e.g., `https://your-project.vercel.app`).

After completing these steps, your application will be live and fully functional in a proper development environment.

---

## 3. Evaluating Other Options

### Can You Use Your Supabase Account?

**Short Answer:** Not without a complete backend rewrite.

Supabase is an excellent platform, but it is a direct alternative to Google's Firebase, not a general-purpose hosting service. It provides its own specific set of backend services, primarily:

*   A managed **PostgreSQL** database.
*   A dedicated **Authentication** service.
*   Auto-generated **REST APIs**.

Your current backend is built with **Python/FastAPI** and **MongoDB**. This stack is fundamentally incompatible with Supabase's architecture. To use Supabase, you would need to:

1.  **Abandon the Python/FastAPI backend** entirely.
2.  **Migrate your database schema** from MongoDB (a NoSQL, document-based database) to PostgreSQL (a SQL, relational database).
3.  **Rewrite all backend logic** (user creation, credit management, Stripe webhooks, referral logic) using Supabase's client libraries, most likely in JavaScript/TypeScript.

**Conclusion:** While Supabase is a powerful tool, it is not a suitable host for your *existing* application. It would be a good choice if you were starting a new project from scratch and wanted to build on its specific ecosystem.

### Can You Use Siteground?

**Short Answer:** It is not recommended. It will be more difficult, less performant, and likely more expensive than the recommended stack.

Siteground's core business is shared hosting for websites, particularly WordPress. These environments are highly optimized for running PHP and are not designed for complex applications like yours.

To make it work, you would need to:

1.  **Upgrade to a VPS/Cloud Hosting Plan:** You cannot run a Python server or host a MongoDB database on their basic shared plans.
2.  **Perform Manual Server Administration:** You would have to use SSH to connect to a bare Linux server and manually install, configure, and maintain all the necessary software:
    *   Python 3.11+
    *   Node.js (to build the React frontend)
    *   MongoDB Server
    *   A process manager (like `systemd` or `supervisor`) to keep your FastAPI server running.
    *   A web server (like Nginx) to act as a reverse proxy.
3.  **Forgo Automated Deployments:** Deploying updates would be a manual process of pulling changes from Git and restarting the server, unlike the seamless, automatic deployments offered by Vercel and Render.

**Conclusion:** Using Siteground would negate many of the benefits of a modern development workflow. It introduces significant complexity and maintenance overhead that is completely handled for you by services like Render and Vercel.

---

## 4. Final Recommendation

For a smooth, scalable, and cost-effective development environment, the **Vercel + Render + MongoDB Atlas** stack is the undisputed best choice for your application. It aligns perfectly with your technology stack, offers superior developer tools, and provides a clear path for scaling from a free development tier to a robust production setup.

Follow the step-by-step guide in Section 2 to get your development environment live.

### References

[1] Vercel. "Vercel Pricing." Accessed November 23, 2025. [https://vercel.com/pricing](https://vercel.com/pricing)

[2] Render. "Render Pricing." Accessed November 23, 2025. [https://render.com/pricing](https://render.com/pricing)

[3] MongoDB. "MongoDB Atlas Pricing." Accessed November 23, 2025. [https://www.mongodb.com/pricing](https://www.mongodb.com/pricing)
