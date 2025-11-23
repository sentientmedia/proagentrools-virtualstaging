# Database Schema Design

## Overview
This schema supports user authentication, credit management, referral tracking, payment processing, and virtual staging operations.

## Tables

### users
Primary user account table with authentication and profile information.

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(255),
  company_name VARCHAR(255),
  phone VARCHAR(50),
  role VARCHAR(50) DEFAULT 'agent', -- 'agent', 'admin'
  
  -- Credits
  credits_balance INTEGER DEFAULT 500, -- Start with 500 signup credits
  total_credits_earned INTEGER DEFAULT 500,
  total_credits_spent INTEGER DEFAULT 0,
  
  -- Referral tracking
  referral_code VARCHAR(20) UNIQUE NOT NULL,
  referred_by_user_id UUID REFERENCES users(id),
  referral_reward_claimed BOOLEAN DEFAULT FALSE,
  
  -- Account status
  email_verified BOOLEAN DEFAULT FALSE,
  account_status VARCHAR(50) DEFAULT 'active', -- 'active', 'suspended', 'deleted'
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_login_at TIMESTAMP,
  
  INDEX idx_email (email),
  INDEX idx_referral_code (referral_code),
  INDEX idx_referred_by (referred_by_user_id)
);
```

### credit_transactions
Track all credit movements (purchases, usage, referrals, bonuses).

```sql
CREATE TABLE credit_transactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  
  -- Transaction details
  transaction_type VARCHAR(50) NOT NULL, -- 'purchase', 'signup_bonus', 'referral_bonus', 'usage', 'refund', 'admin_adjustment'
  amount INTEGER NOT NULL, -- Positive for credits added, negative for credits spent
  balance_after INTEGER NOT NULL,
  
  -- Reference tracking
  related_payment_id UUID REFERENCES payments(id),
  related_staging_id UUID REFERENCES staging_jobs(id),
  related_referral_user_id UUID REFERENCES users(id),
  
  -- Metadata
  description TEXT,
  metadata JSONB, -- Flexible field for additional data
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_user_id (user_id),
  INDEX idx_transaction_type (transaction_type),
  INDEX idx_created_at (created_at)
);
```

### payments
Track all payment transactions for credit purchases.

```sql
CREATE TABLE payments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  
  -- Payment details
  amount_usd DECIMAL(10, 2) NOT NULL,
  credits_purchased INTEGER NOT NULL,
  payment_status VARCHAR(50) NOT NULL, -- 'pending', 'completed', 'failed', 'refunded'
  
  -- Payment gateway integration
  payment_provider VARCHAR(50) NOT NULL, -- 'stripe', 'paypal', etc.
  payment_provider_transaction_id VARCHAR(255),
  payment_method VARCHAR(50), -- 'card', 'bank_transfer', etc.
  
  -- Pricing tier
  pricing_tier VARCHAR(50), -- 'starter', 'professional', 'enterprise'
  
  -- Metadata
  metadata JSONB,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  
  INDEX idx_user_id (user_id),
  INDEX idx_payment_status (payment_status),
  INDEX idx_provider_transaction_id (payment_provider_transaction_id)
);
```

### staging_jobs
Track all virtual staging operations.

```sql
CREATE TABLE staging_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  
  -- Job details
  job_status VARCHAR(50) NOT NULL, -- 'pending', 'processing', 'completed', 'failed'
  credits_used INTEGER DEFAULT 1,
  
  -- Input parameters
  prompt TEXT NOT NULL,
  strength DECIMAL(3, 2) DEFAULT 0.75,
  guidance_scale DECIMAL(4, 2) DEFAULT 7.5,
  
  -- File storage
  input_image_url TEXT NOT NULL,
  output_image_url TEXT,
  
  -- Replicate integration
  replicate_prediction_id VARCHAR(255),
  replicate_status VARCHAR(50),
  
  -- Processing metrics
  processing_time_seconds INTEGER,
  error_message TEXT,
  
  -- Metadata
  metadata JSONB,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  
  INDEX idx_user_id (user_id),
  INDEX idx_job_status (job_status),
  INDEX idx_created_at (created_at),
  INDEX idx_replicate_prediction_id (replicate_prediction_id)
);
```

### referrals
Track referral relationships and reward status.

```sql
CREATE TABLE referrals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  referrer_user_id UUID NOT NULL REFERENCES users(id),
  referred_user_id UUID NOT NULL REFERENCES users(id),
  
  -- Referral status
  referral_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'qualified', 'rewarded'
  
  -- Reward tracking
  reward_credits INTEGER DEFAULT 500,
  reward_granted_at TIMESTAMP,
  qualifying_payment_id UUID REFERENCES payments(id),
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_referrer_user_id (referrer_user_id),
  INDEX idx_referred_user_id (referred_user_id),
  INDEX idx_referral_status (referral_status),
  
  UNIQUE(referrer_user_id, referred_user_id)
);
```

### pricing_tiers
Define credit packages and pricing.

```sql
CREATE TABLE pricing_tiers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  tier_name VARCHAR(100) NOT NULL,
  tier_slug VARCHAR(50) UNIQUE NOT NULL,
  
  -- Pricing
  credits INTEGER NOT NULL,
  price_usd DECIMAL(10, 2) NOT NULL,
  price_per_credit DECIMAL(10, 4) GENERATED ALWAYS AS (price_usd / credits) STORED,
  
  -- Display
  display_order INTEGER DEFAULT 0,
  is_featured BOOLEAN DEFAULT FALSE,
  is_active BOOLEAN DEFAULT TRUE,
  
  -- Features
  description TEXT,
  features JSONB, -- Array of feature descriptions
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  INDEX idx_tier_slug (tier_slug),
  INDEX idx_display_order (display_order)
);
```

### api_keys
For future API access (optional).

```sql
CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id),
  
  key_name VARCHAR(255) NOT NULL,
  api_key_hash VARCHAR(255) NOT NULL,
  api_key_prefix VARCHAR(20) NOT NULL, -- First few chars for identification
  
  -- Permissions
  permissions JSONB, -- Array of allowed operations
  
  -- Status
  is_active BOOLEAN DEFAULT TRUE,
  last_used_at TIMESTAMP,
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP,
  
  INDEX idx_user_id (user_id),
  INDEX idx_api_key_prefix (api_key_prefix)
);
```

## Initial Data

### Pricing Tiers (Example)

```sql
INSERT INTO pricing_tiers (tier_name, tier_slug, credits, price_usd, display_order, is_featured, description, features) VALUES
('Starter Pack', 'starter', 50, 9.99, 1, false, 'Perfect for trying out virtual staging', '["50 staging credits", "Standard processing", "Email support"]'::jsonb),
('Professional', 'professional', 200, 29.99, 2, true, 'Best value for active agents', '["200 staging credits", "Priority processing", "Email & chat support", "15% savings"]'::jsonb),
('Enterprise', 'enterprise', 500, 59.99, 3, false, 'For high-volume teams', '["500 staging credits", "Fastest processing", "Priority support", "25% savings", "API access"]'::jsonb),
('Ultimate', 'ultimate', 1000, 99.99, 4, false, 'Maximum value package', '["1000 staging credits", "Fastest processing", "Dedicated support", "30% savings", "API access", "Custom branding"]'::jsonb);
```

## Indexes and Performance

Key indexes for common queries:
- User lookups by email and referral code
- Credit transaction history by user and date
- Staging job history by user and status
- Referral tracking and reward status
- Payment history and status

## Security Considerations

1. **Password Security**: Use bcrypt or Argon2 for password hashing
2. **API Keys**: Hash API keys, store only prefix for display
3. **PII Protection**: Encrypt sensitive user data at rest
4. **Audit Logging**: Track all credit transactions and payments
5. **Rate Limiting**: Implement at application layer based on user tier

## Migration Strategy

1. Create tables in order of dependencies
2. Add indexes after initial data load
3. Set up foreign key constraints
4. Create database triggers for updated_at timestamps
5. Set up backup and replication
