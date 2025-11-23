# Referral System Flow Diagram

## Visual Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     REFERRAL SYSTEM FLOW                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   USER A     │
│  (Referrer)  │
└──────┬───────┘
       │
       │ 1. Signs up
       ▼
┌─────────────────────────────┐
│  ✅ Gets 500 credits         │
│  ✅ Gets referral code: XYZ  │
│  ✅ total_referrals = 0      │
└─────────────────────────────┘
       │
       │ 2. Shares referral link
       │    /register?ref=XYZ
       ▼
┌──────────────┐
│   USER B     │
│  (Referred)  │
└──────┬───────┘
       │
       │ 3. Signs up with code XYZ
       ▼
┌────────────────────────────────────────┐
│  ✅ Gets 500 credits                    │
│  ✅ referred_by = User A's ID           │
│  ✅ first_purchase_completed = false    │
│                                         │
│  User A:                                │
│  ✅ total_referrals = 1                 │
│  ❌ NO bonus credits yet                │
└────────────────────────────────────────┘
       │
       │ 4. User B uses the platform
       │    (virtual staging, etc.)
       │
       │ 5. User B runs low on credits
       ▼
┌──────────────────────────┐
│  User B buys credits     │
│  (First Purchase!)       │
└──────┬───────────────────┘
       │
       │ 6. Stripe payment completes
       ▼
┌────────────────────────────────────────────────────────┐
│               WEBHOOK TRIGGERED                         │
│  checkout.session.completed                             │
│                                                         │
│  Backend checks:                                        │
│  ✅ Is this User B's first purchase?                    │
│  ✅ Does User B have a referrer (User A)?               │
│                                                         │
│  If YES to both:                                        │
│  ┌──────────────────────────────────────────┐          │
│  │ 1. Add purchased credits to User B       │          │
│  │ 2. Set first_purchase_completed = true   │          │
│  │ 3. Award 500 credits to User A           │          │
│  │ 4. Log referral_bonus transaction        │          │
│  └──────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│          FINAL STATE                     │
│                                          │
│  User A (Referrer):                      │
│  ✅ +500 bonus credits                   │
│  ✅ total_referrals = 1                  │
│  ✅ Can see User B in referral list      │
│     with "Completed" status              │
│                                          │
│  User B (Referred):                      │
│  ✅ Credits from purchase added          │
│  ✅ first_purchase_completed = true      │
│  ✅ Can now refer others                 │
└─────────────────────────────────────────┘
```

## Key Points

### ✅ Credits Awarded

| Event | User A (Referrer) | User B (Referred) |
|-------|-------------------|-------------------|
| User A signs up | 500 | - |
| User B signs up with A's code | 0 | 500 |
| User B makes first purchase | **+500 BONUS** | +Purchased amount |
| User B makes second purchase | 0 | +Purchased amount |

### 🔄 Database Changes

#### On User B Signup:
```javascript
// User B document
{
  credits: 500,
  referred_by: "user_a_id",
  first_purchase_completed: false
}

// User A document
{
  total_referrals: 1  // incremented
}
```

#### On User B First Purchase:
```javascript
// User B document
{
  credits: 500 + purchased_amount,
  first_purchase_completed: true  // changed!
}

// User A document
{
  credits: original + 500  // bonus added!
}

// New transaction record
{
  type: "referral_bonus",
  user_id: "user_a_id",
  amount: 500,
  description: "Referral bonus: User B made their first purchase"
}
```

### 🎯 Viral Loop Mechanics

```
User A → Refers 5 friends → All 5 make purchases
Result: User A earns 2,500 credits (5 × 500)

Each of those 5 can refer more users...
Exponential growth! 📈
```

### 🚫 What Does NOT Trigger Bonus

- User B signing up (only tracks, doesn't pay)
- User B using free credits
- User B making 2nd, 3rd, etc. purchases
- User A referring someone who never purchases

### ✅ What DOES Trigger Bonus

- User B making their FIRST purchase
- Only triggers ONCE per referred user
- Works regardless of purchase amount

## Implementation Details

### Backend Logic (Simplified)

```python
# In webhook handler
if event['type'] == 'checkout.session.completed':
    user = get_user(user_id)
    
    # Check if first purchase
    if not user.first_purchase_completed:
        # Mark as completed
        user.first_purchase_completed = True
        
        # Award referrer if exists
        if user.referred_by:
            referrer = get_user(user.referred_by)
            referrer.credits += 500
            log_transaction(
                user_id=referrer.id,
                type="referral_bonus",
                amount=500
            )
```

### Frontend Display

```javascript
// In ReferralDashboard
{referrals.map(referral => (
  <div>
    {referral.first_purchase_completed ? (
      <Badge color="green">Completed - 500 credits earned</Badge>
    ) : (
      <Badge color="yellow">Pending first purchase</Badge>
    )}
  </div>
))}
```

## Testing Scenarios

### Scenario 1: Happy Path ✅
1. Alice signs up → 500 credits
2. Bob signs up with Alice's code → 500 credits
3. Bob buys $25 package → Bob gets 300, Alice gets 500 bonus
4. ✅ Alice sees Bob as "Completed" in referral list

### Scenario 2: Multiple Referrals ✅
1. Alice refers Bob, Carol, Dave
2. All 3 sign up → Alice total_referrals = 3
3. Only Bob buys credits → Alice gets 500 (not 1500)
4. Later Carol buys → Alice gets another 500
5. Dave never buys → Alice never gets bonus for Dave

### Scenario 3: Second Purchase ✅
1. Bob already made first purchase
2. Bob buys again → Bob gets credits
3. Alice gets nothing (already got bonus)

### Scenario 4: Self-Referral ❌
1. Alice tries to use her own referral code
2. System should prevent this (add validation if needed)

---

**Key Takeaway:** The referral bonus is tied to FIRST PURCHASE, creating a strong incentive for referrers to help new users succeed on the platform! 🎯
