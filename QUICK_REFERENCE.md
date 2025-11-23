# ProAgentTools - Quick Reference Card

## 🎯 New Features at a Glance

### Credit System
- **Signup Bonus:** 500 credits (was 100)
- **Packages:** $10 (100), $25 (300), $50 (700), $100 (1500)
- **Purchase:** Click credit button in header → Select package → Stripe checkout

### Referral System
- **Your Code:** Found in `/referrals` dashboard
- **Bonus:** 500 credits when referred user makes FIRST PURCHASE
- **Share:** Email, Twitter, LinkedIn buttons available

### New Pages
| URL | Purpose |
|-----|---------|
| `/credits/purchase` | Buy credit packages |
| `/credits/transactions` | View transaction history |
| `/payment/success` | Payment confirmation |
| `/payment/cancel` | Cancelled payment |
| `/referrals` | Referral dashboard & stats |

### New API Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/credits/packages` | List packages |
| POST | `/api/credits/create-checkout-session` | Start purchase |
| POST | `/api/credits/webhook` | Stripe webhook |
| GET | `/api/credits/transactions` | Transaction history |
| GET | `/api/referrals/stats` | Referral statistics |
| GET | `/api/referrals/list` | List referrals |

## 🔧 Quick Setup

### 1. Stripe Configuration
```bash
# In Stripe Dashboard:
# 1. Get API keys (Developers → API keys)
# 2. Create 4 products with prices
# 3. Set up webhook endpoint
# 4. Copy webhook secret
```

### 2. Environment Variables
```bash
# Backend .env
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_STARTER=price_...
STRIPE_PRICE_PRO=price_...
STRIPE_PRICE_BUSINESS=price_...
STRIPE_PRICE_ENTERPRISE=price_...
FRONTEND_URL=https://your-domain.com

# Frontend .env
REACT_APP_API_URL=https://api.your-domain.com
```

### 3. Database Migration
```javascript
// MongoDB
db.users.updateMany(
  { first_purchase_completed: { $exists: false } },
  { $set: { first_purchase_completed: false, credits: 500 } }
);
```

## 🧪 Testing Flow

### Test Referral System
1. User A signs up → Gets 500 credits + referral code
2. User B signs up with A's code → Gets 500 credits
3. User B buys credits → A gets 500 bonus credits ✅

### Test Payment
1. Go to `/credits/purchase`
2. Select package
3. Use test card: `4242 4242 4242 4242`
4. Verify credits added
5. Check `/credits/transactions`

## 📊 Key Metrics

### MongoDB Queries
```javascript
// Total revenue
db.credit_transactions.aggregate([
  { $match: { type: "purchase" } },
  { $group: { _id: null, total: { $sum: "$amount" } } }
]);

// Top referrers
db.users.find().sort({ total_referrals: -1 }).limit(10);

// Conversion rate
db.users.aggregate([
  { $match: { referred_by: { $ne: null } } },
  { $group: {
      _id: null,
      total: { $sum: 1 },
      converted: { $sum: { $cond: ["$first_purchase_completed", 1, 0] } }
    }
  }
]);
```

## 🐛 Troubleshooting

### Webhook Not Working
- Check URL in Stripe Dashboard
- Verify webhook secret matches .env
- Look at Stripe Dashboard → Webhooks → Events

### Credits Not Added
- Check webhook logs
- Verify `checkout.session.completed` event
- Check MongoDB for transaction record

### Referral Bonus Not Triggered
- Verify `first_purchase_completed` field exists
- Check `referred_by` is set
- Look for `referral_bonus` transaction

## 📞 Quick Links
- [Full Implementation Guide](./IMPLEMENTATION_GUIDE.md)
- [Changes Summary](./CHANGES_SUMMARY.md)
- [Stripe Docs](https://stripe.com/docs)
- [Stripe Test Cards](https://stripe.com/docs/testing)

---

**Remember:** Referral bonus = 500 credits on FIRST PURCHASE, not signup! 🎉
