# Railway Deployment Guide for ProAgentTools

## Overview

Your repository has a monorepo structure with multiple services:
- `/backend` - Python/FastAPI backend
- `/frontend` - React frontend
- Root directory - AI model files (not needed for the web app)

Railway needs to deploy these as **separate services** from the same repository.

---

## Step 1: Create MongoDB Database on Railway

1. **Sign up/Login to Railway:** Go to [railway.app](https://railway.app) and sign in with GitHub

2. **Create a New Project:**
   - Click "New Project"
   - Select "Deploy MongoDB"
   - Railway will provision a MongoDB instance

3. **Get Connection String:**
   - Click on your MongoDB service
   - Go to "Variables" tab
   - Copy the `MONGO_URL` value (looks like `mongodb://mongo:...`)
   - Save this for later

---

## Step 2: Deploy Backend Service

1. **Add New Service:**
   - In your Railway project, click "New"
   - Select "GitHub Repo"
   - Choose `sentientmedia/proagentrtools-virtualstaging`
   - Select branch: `conflict_201125_1312`

2. **Configure Root Directory:**
   - After the service is created, click on it
   - Go to "Settings" tab
   - Scroll to "Service Settings"
   - Set **Root Directory** to: `backend`
   - Click "Update"

3. **Add Environment Variables:**
   - Go to "Variables" tab
   - Click "New Variable" and add each of these:

   ```
   MONGO_URL=<your MongoDB connection string from Step 1>
   DB_NAME=proagentrtools
   SECRET_KEY=<generate a random secret key>
   REPLICATE_API_TOKEN=<your replicate token>
   RUNPOD_API_KEY=<your runpod key>
   OPENAI_API_KEY=<your openai key>
   STRIPE_SECRET_KEY=<your stripe secret key>
   STRIPE_WEBHOOK_SECRET=<your stripe webhook secret>
   STRIPE_PRICE_STARTER=<price_id from Stripe>
   STRIPE_PRICE_PRO=<price_id from Stripe>
   STRIPE_PRICE_BUSINESS=<price_id from Stripe>
   STRIPE_PRICE_ENTERPRISE=<price_id from Stripe>
   FRONTEND_URL=<will add after frontend is deployed>
   PORT=8000
   ```

4. **Deploy:**
   - Railway will automatically detect Python and deploy
   - Wait for deployment to complete
   - Copy your backend URL (e.g., `https://proagent-backend-production.up.railway.app`)

---

## Step 3: Deploy Frontend Service

1. **Add Another Service:**
   - In the same Railway project, click "New"
   - Select "GitHub Repo"
   - Choose the same repository: `sentientmedia/proagentrtools-virtualstaging`
   - Select branch: `conflict_201125_1312`

2. **Configure Root Directory:**
   - Click on the new service
   - Go to "Settings" tab
   - Set **Root Directory** to: `frontend`
   - Click "Update"

3. **Add Environment Variables:**
   - Go to "Variables" tab
   - Add these variables:

   ```
   REACT_APP_BACKEND_URL=<your backend URL from Step 2>
   REACT_APP_API_URL=<your backend URL from Step 2>
   ```

4. **Deploy:**
   - Railway will detect React and build it
   - Wait for deployment
   - Copy your frontend URL (e.g., `https://proagent-frontend-production.up.railway.app`)

---

## Step 4: Update Backend with Frontend URL

1. **Go back to Backend Service:**
   - Click on your backend service
   - Go to "Variables" tab
   - Find `FRONTEND_URL` variable
   - Update it with your frontend URL from Step 3
   - The backend will automatically redeploy

---

## Step 5: Configure Stripe Webhook

1. **Go to Stripe Dashboard:**
   - Navigate to Developers → Webhooks
   - Click "Add endpoint"

2. **Set Webhook URL:**
   - Endpoint URL: `<your-backend-url>/api/credits/webhook`
   - Example: `https://proagent-backend-production.up.railway.app/api/credits/webhook`

3. **Select Events:**
   - Click "Select events"
   - Choose: `checkout.session.completed`
   - Click "Add events"

4. **Get Webhook Secret:**
   - After creating, copy the "Signing secret" (starts with `whsec_`)
   - Update the `STRIPE_WEBHOOK_SECRET` variable in your Railway backend service

---

## Step 6: Test Your Deployment

1. **Visit Frontend URL:**
   - Open your frontend URL in a browser
   - You should see the ProAgentTools landing page

2. **Test Signup:**
   - Click "Sign In" and create a new account
   - Verify you receive 500 credits

3. **Test Credit Purchase:**
   - Navigate to the credit purchase page
   - Try buying credits with Stripe test card: `4242 4242 4242 4242`
   - Verify credits are added

4. **Test Referral:**
   - Go to the referral dashboard
   - Copy your referral link
   - Sign up a second user with the referral code
   - Have that user make a purchase
   - Verify you receive 500 bonus credits

---

## Troubleshooting

### Backend Won't Start

**Check Logs:**
- Go to backend service → "Deployments" tab
- Click on the latest deployment
- Check the logs for errors

**Common Issues:**
- Missing environment variables
- Wrong MongoDB connection string
- Port not set correctly (should be `$PORT` in start command)

### Frontend Shows "Cannot connect to backend"

**Check:**
- `REACT_APP_BACKEND_URL` is set correctly in frontend variables
- Backend service is running (check status)
- CORS is configured in backend (already done in server.py)

### Webhook Not Working

**Verify:**
- Webhook URL matches your backend URL exactly
- Webhook secret matches the one in Railway variables
- Event type is `checkout.session.completed`
- Check Stripe Dashboard → Webhooks → Events for delivery status

---

## Railway Project Structure

After deployment, your Railway project should have:

```
ProAgentTools Project
├── MongoDB (Database)
├── Backend Service (Python/FastAPI)
│   └── Root Directory: backend/
└── Frontend Service (React)
    └── Root Directory: frontend/
```

---

## Cost Estimate

**Development (Low Traffic):**
- MongoDB: ~$2-3/month
- Backend: ~$2-3/month
- Frontend: ~$1-2/month
- **Total: ~$5-8/month**

**Production (Moderate Traffic):**
- Scales automatically based on usage
- Monitor in Railway dashboard

---

## Next Steps

1. ✅ Deploy all services
2. ✅ Configure environment variables
3. ✅ Set up Stripe webhook
4. ✅ Test the complete flow
5. 🎯 Launch and start getting users!

Your viral loop is ready to go! 🚀
