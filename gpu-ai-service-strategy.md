# GPU-Powered AI Service: Business Strategy & Pricing Plan

**Date:** 2026
**Target:** Compete with Claude for unlimited, affordable AI coding assistance
**Core Model:** Qwen3.5-122B-A10B (MoE) on Vast.ai A100 80GB

---

## Executive Summary

Building a GPU rental AI service that spins up Vast.ai instances for unlimited token access to open-source models. The key differentiator is **unlimited usage without token caps** at prices 60% cheaper than Claude for power users.

**Key Insight:** Qwen3.5-122B-A10B uses MoE architecture (10B active parameters out of 122B total), fitting on a single A100 80GB at ~75GB VRAM with NVFP4 quantization. This dramatically reduces infrastructure costs.

---

## Actual Vast.ai GPU Pricing (2025-2026)

| GPU | Vast.ai Price | Notes |
|-----|---------------|-------|
| RTX 4090 (24GB) | $0.14-0.35/hr | Average ~$0.25/hr |
| A100 80GB | $0.72-1.40/hr | Average ~$0.85/hr |
| H100 PCIe (80GB) | $1.50-2.00/hr | Premium tier |

---

## The Game-Changer: Qwen3.5-122B-A10B MoE Architecture

### Key Facts
- **122B total parameters** but only **10B active per token** (Mixture of Experts)
- **NVFP4 quantization:** ~75GB VRAM needed
- **Fits on single A100 80GB** ✅
- **Optimized for:** Coding, reasoning, visual understanding
- **256 sparse experts** for high capability at lower compute

### Why This Matters
This means your cost structure is MUCH better than traditional 122B models would suggest. You don't need dual GPUs - single A100 80GB is sufficient.

---

## Revised Cost Analysis (Single A100 80GB)

| Component | Cost |
|-----------|------|
| **Vast.ai A100 80GB** | $0.72-1.40/hr (avg $0.85/hr) |
| **Bandwidth/overhead** | ~$0.05/hr |
| **Total cost** | **~$0.90/hr** |

---

## Competitive Positioning vs Claude

### Claude Pricing (2025)

| Plan | Price | Usage Limits |
|------|-------|--------------|
| **Free** | $0 | ~10-15 messages/session |
| **Pro** | $20/mo | 5x Free limits (~50-75 msgs/session) |
| **Max** | $100-200/mo | 20x Free limits (~200-300 msgs/session) |
| **API** | $3-25 per million tokens | Pay-as-you-go |

### Heavy Developer Usage on Claude
- 50K tokens/day = ~$15-50/day = **$450-1,500/month**
- **You can undercut this dramatically**

### Your Advantage
| Feature | Claude | Your Service |
|---------|--------|--------------|
| Token limits | Yes (strict) | **Unlimited** |
| Model choice | Fixed (Claude 3.5) | **Multiple open-source models** |
| Privacy | Cloud-hosted | **Self-hosted GPU** |
| Price (heavy use) | $450-1,500/mo | **$150-300/mo** |
| IDE integration | Web-only | **VS Code extension** |

---

## Pricing Tiers & Models

### Tier 1: 122B MoE (Flagship)
- **Cost:** $0.90/hr
- **Price:** **$1.50-2.00/hr** (or $30-40/month unlimited)
- **Margin:** $0.60-1.10/hr (40-55%)
- **Positioning:** "Claude-quality coding, unlimited, 60% cheaper"

### Tier 2: 70B Models (Backup/Speed)
- **Cost:** $0.85/hr (single A100)
- **Price:** $1.20-1.50/hr
- **Margin:** $0.35-0.65/hr (30-45%)
- **Use case:** Faster inference, lighter tasks

### Tier 3: Smaller Models (Quick Tasks)
- **Cost:** $0.25-0.50/hr (RTX 4090)
- **Price:** $0.60-0.90/hr
- **Margin:** $0.35-0.40/hr (50-60%)
- **Use case:** Code snippets, quick Q&A

---

## Pricing Models to Consider

### Option A: Hourly (Pay-as-you-go)

| Model | Price | Your Cost | Margin |
|-------|-------|-----------|--------|
| 122B MoE | $1.80/hr | $0.90/hr | 50% |
| 70B | $1.30/hr | $0.85/hr | 35% |
| 13B | $0.70/hr | $0.27/hr | 61% |

**Break-even:** 10 concurrent 122B users = $18/hr revenue, $9/hr cost = **$9/hr profit**

### Option B: Monthly Subscription (Developer Plans)

| Plan | Price | Features | Target User |
|------|-------|----------|-------------|
| **Starter** | $29/mo | 122B, 10 hrs/month | Hobbyists |
| **Pro** | $79/mo | 122B, 40 hrs/month | Freelancers |
| **Unlimited** | $149/mo | 122B, unlimited | Power users |
| **Team** | $299/mo | 5 seats, shared pool | Small teams |

**Math:** If 1,000 Pro users pay $79/mo = **$79,000/month revenue**
- Average usage: 20 hrs/user = 20,000 hrs/month
- GPU cost: 20,000 hrs × $0.90 = $18,000
- **Profit: $61,000/month** (77% margin)

### Option C: Hybrid (Recommended)
- **Pay-as-you-go:** $1.80/hr for 122B
- **Credits:** Buy $50 get $60 bonus
- **Subscription:** Unlimited 122B for $149/mo
- **API access:** $0.05 per 1K tokens for integrations

---

## Developer-Focused Features That Sell

### 1. "Unlimited Coding Sessions"
- No token caps
- No message limits
- No rate throttling during peak hours
- **Target:** Developers who hit Claude's $200/mo ceiling

### 2. IDE Integration
- VS Code extension
- Cursor-like experience
- Direct code generation, debugging, refactoring
- **Differentiation:** Claude requires web interface

### 3. Project Context Memory
- Upload entire codebases
- Model remembers project structure
- "Scrolls" = pre-built coding workflows
- **Example:** "React project setup," "Python API scaffold"

### 4. Swarm Mode for Complex Tasks
- Parallel model instances
- Break down large refactors
- Test multiple approaches simultaneously
- **Premium feature:** +50% pricing

### 5. Model Switching
- Start with 122B for architecture
- Switch to 70B for quick edits
- Use smaller models for snippets
- **All in one session**

---

## Go-to-Market for Developers

### Target Audiences
1. **Freelance developers** - Hit Claude's limits, need cost predictability
2. **Startup CTOs** - Building AI-powered tools, need API access
3. **Open source contributors** - Testing models, building with AI
4. **Enterprise teams** - Privacy concerns, want self-hosted option

### Marketing Angles
- **"Claude-quality coding, 60% cheaper"**
- **"Unlimited tokens, no surprises"**
- **"Your codebase, your model, your privacy"**
- **"Build AI features without the API bill shock"**

### Launch Strategy
1. **Beta:** Invite 100 developers (free unlimited for 30 days)
2. **Feedback loop:** Improve model selection, IDE integration
3. **Public launch:** $1.80/hr or $149/mo unlimited
4. **Partner:** VS Code marketplace, GitHub sponsors

---

## Profit Projections (Conservative)

| Month | Users | Avg Usage | Revenue | GPU Cost | Profit |
|-------|-------|-----------|---------|----------|--------|
| 1 | 50 | 15 hrs | $1,350 | $675 | $675 |
| 3 | 200 | 20 hrs | $7,200 | $3,600 | $3,600 |
| 6 | 500 | 25 hrs | $22,500 | $11,250 | $11,250 |
| 12 | 1,500 | 30 hrs | $81,000 | $40,500 | $40,500 |

**Key insight:** Even at 50% margin, you're profitable from day 1.

---

## Technical Architecture Recommendations

### 1. GPU Pool Management
- Pre-provision 5-10 A100 instances
- Auto-scale based on demand
- Queue system for peak times

### 2. Model Caching
- Keep 122B MoE loaded (75GB fits)
- Hot-swap between 70B/13B models
- Reduce spin-up time to <30 seconds

### 3. Session Persistence
- Save context between requests
- Project-level memory
- "Scrolls" = saved workflows

### 4. Multi-tenant Support
- Run multiple users on same GPU
- Isolated contexts
- Fair usage limits per session

### 5. Idle Timeout
- Turn off after 10 min of inactivity
- Save session state to disk
- Quick restore on reconnect

---

## Risk Mitigation

1. **GPU Availability** - Have backup providers (Lambda, RunPod)
2. **Model Costs** - Cache popular models to reduce spin-up time
3. **Payment Fraud** - Prepaid credits or card verification
4. **Customer Support** - Automated + community-driven to keep costs low
5. **Rate Limits** - Fair usage policy to prevent abuse

---

## Additional Revenue Streams

1. **Model Marketplace** - Curate and sell access to specialized fine-tuned models (+10-20% markup)
2. **API Access** - Developer API at $0.05-0.10 per 1K tokens
3. **Team Plans** - Multiple seats, shared billing, usage analytics
4. **Scrolls/Skills Templates** - Pre-built workflows for specific use cases
5. **Swarm Mode Premium** - Parallel processing for complex tasks (+50% premium)

---

## Recommended Pricing Strategy

### Initial Launch (Months 1-3)
- **122B MoE:** $1.50/hr (aggressive to capture market)
- **70B:** $1.00/hr
- **13B:** $0.50/hr

### Growth Phase (Months 4-12)
- **122B MoE:** $1.80/hr
- **70B:** $1.30/hr
- **13B:** $0.70/hr

### Subscription Tiers (Always Available)
- **Starter:** $29/mo (10 hrs 122B)
- **Pro:** $79/mo (40 hrs 122B)
- **Unlimited:** $149/mo (unlimited 122B)

### Minimum Session Blocks
- **15-minute minimum** to cover spin-up costs
- **No charge for first 5 minutes** (trial period)

---

## Key Success Metrics

1. **Concurrent Users:** Target 50+ by month 3, 200+ by month 6
2. **Average Session Duration:** Target 45+ minutes
3. **Churn Rate:** Keep under 10% monthly
4. **GPU Utilization:** Target 60%+ during business hours
5. **Customer Acquisition Cost:** Keep under $50 per paying user

---

## Bottom Line

**Your 122B MoE model on single A100 is the killer app:**
- ✅ Fits on affordable hardware ($0.85/hr)
- ✅ Claude-quality coding performance
- ✅ Unlimited usage (no token caps)
- ✅ 50%+ margins at $1.80/hr
- ✅ $149/mo unlimited beats Claude's $200/mo with limits

**You're not just competing - you're disrupting.**

---

## Next Steps

1. **Build MVP:** Single A100 instance with 122B MoE
2. **Create VS Code extension:** Basic code generation interface
3. **Set up billing:** Stripe integration with prepaid credits
4. **Beta test:** Invite 100 developers for feedback
5. **Launch:** Public release with aggressive pricing

---

*Document created for GPU-powered AI service business planning*
*All pricing based on Vast.ai marketplace rates as of 2025-2026*
