# Mandatory zero-cost provider policy

Policy review / repository audit date: **2026-10-09**.
Project: ACTIVE. Phase 5: IN PROGRESS. Next candidate: Mistral direct.

This records the user's reviewed external-policy input and repository findings.
No provider, pricing/catalog, account/billing or inference API was called during
this audit. The date is a product-policy/repository check date, not a fresh remote
catalog verification date. Never infer account billing state from an API key or
a model name. Operator evidence must be checked before live use.

## ZERO-COST GATE: before adapter implementation

A production or live evaluation provider/model is eligible only when all eight
conditions hold:

1. The selected API/model can operate at $0.
2. That free mode requires no payment method or credit card.
3. It cannot automatically incur paid overage.
4. Exhausted free quota fails closed rather than charging or upgrading.
5. The selected model is included in that free mode.
6. Temporary trial/promotional credits alone do not qualify.
7. Provider evidence supports free eligibility and account/quota behavior.
8. Eligibility and availability can be rechecked later.

No prepaid credits, paid balance, pay-as-you-go, paid subscription or automatic
overage is allowed. Paid candidates are rejected before adapter implementation;
passing the gate does not itself admit a model to routing. Quotas/rate limits are
acceptable only in a non-billable free account/mode.

Operator evidence should record provider documentation/catalog location, exact
model, Free account mode, no-payment requirement, over-quota behavior, check date
and reviewer. Do not record keys, identity, account IDs, balances or private URLs.
Recheck before live evaluation/deployment, after any account/model/terms change,
and during periodic operational review. Remove eligibility when evidence expires
or contradicts the policy. No runtime pricing fetch or automatic billing setup.

## Eligibility semantics and runtime boundary

- **ELIGIBLE_FREE**: a reviewed free route/model under the required non-billable
  free account mode; not a claim about the operator's actual account.
- **CONDITIONAL_FREE**: provider can be retained only with verified model/account
  and quota conditions. Until confirmed, live use is closed.
- **NOT_VERIFIED**: insufficient model/free-account evidence; do not use live.
- **REJECTED_PAID**: fails the product gate; do not evaluate live or route.

`app/llm/cost_policy.py` is a small offline enum/allow-list guard. It is not a
Phase 6 policy engine, adaptive router, pricing service or account inspector.
Registry checks every configured task model before client construction. Unknown
optional model configurations are disabled; an ineligible primary fails startup
with a controlled configuration error. Existing task priority lists are unchanged.
Evaluation checks selected model eligibility before registry creation and before
each call, including programmatic wrappers. Dual live opt-in and CI prohibition
remain necessary and cannot override rejection or missing cost evidence.

`CLOUDFLARE_ZERO_COST_VERIFIED=false` and `LLM7_ZERO_COST_VERIFIED=false` are safe
defaults. Set the respective flag true only after recording evidence that the
selected allowed model/account is non-billable, needs no payment method and
fails closed at quota exhaustion. These flags attest reviewed conditions; they
cannot disable billing on a provider account or verify remote state themselves.
Groq/OpenRouter/Gemini likewise must use their reviewed non-billable free modes;
their free-model allow-list does not make a paid account free.

Low-level retained adapter classes are implementation/test artifacts, not an
admission bypass. Direct scripts outside registry/evaluation guards must comply
with this policy; do not call rejected or unverified adapters live. Mock/fixture
tests are network-free and do not constitute live admission evidence.

## Current providers

All entries below have policy/repository check date **2026-10-09**. Account mode
was not inspected, and keys/.env were not read. Official portals are future
revalidation locations, not claims of fresh catalog verification in this audit.

### Groq — KEEP / ELIGIBLE_FREE

- Allowed models: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`; user-reviewed Free
  Plan limits apply. Other names remain NOT_VERIFIED until reviewed.
- Free quota: Free Plan rate/quota limits must return a safe rate-limit failure.
- Billing risk: paid/Flex account modes do not qualify; no automatic overage.
- Fail closed: no upgrade, paid-model substitution or balance purchase.
- Evidence: reviewed product input plus repository fast/reasoning defaults.
- Revalidation: selected-model Free Plan limits and non-billable account mode,
  using the [Groq console documentation](https://console.groq.com/docs).

### OpenRouter — KEEP / CONDITIONAL_FREE

- Allowed route/model policy: `openrouter/free` or explicit `:free` routes.
  Unsuffixed $0 models require separately reviewed catalog evidence; the current
  conservative allow-list does not accept them automatically.
- Free quota: failures must remain within free routes, even if paid models exist.
- Billing risk: paid variants, purchased credits and auto-top-up do not qualify.
- Fail closed: never strip `:free`, choose a paid router or buy credits.
- Evidence: reviewed product input and `openrouter/free` configured default.
- Revalidation: current zero input/output pricing, no-payment free eligibility
  and quota behavior via [OpenRouter documentation](https://openrouter.ai/docs).

### Gemini — KEEP / ELIGIBLE_FREE

- Current allowed model: `gemini-3.8-flash`, explicitly accepted as Gemini Free
  Tier eligible by the reviewed product requirement.
- Free quota: use Free Tier; exhaustion must fail, not enter billed usage.
- Billing risk: paid project/tier or paid-only model selection is prohibited.
- Fail closed: unknown model names are blocked, with no automatic tier change.
- Evidence: reviewed product input and current model setting; no fresh remote
  verification is claimed.
- Revalidation: exact model Free Tier availability and account/project mode via
  [Gemini API documentation](https://ai.google.dev/gemini-api/docs).

### Cloudflare — KEEP WITH GUARD / CONDITIONAL_FREE

- Allowed implementation profile: configured `cf/openai/gpt-oss-120b` only after
  the operator verifies this exact model is covered by Workers Free allocation.
  This model's unconditional free eligibility is not asserted by the repository.
- Free quota: allocation exhaustion must fail, with no billed continuation.
- Billing risk: Workers Paid accounts, paid-only models and overage charges.
- Fail closed: registry disables this provider until
  `CLOUDFLARE_ZERO_COST_VERIFIED=true`; unknown models remain blocked even then.
- Evidence: user-reviewed Workers Free policy; model/account allocation evidence
  is still an operator precondition, not established by token/account ID presence.
- Revalidation: account mode, model allocation inclusion and over-quota behavior
  via [Workers AI documentation](https://developers.cloudflare.com/workers-ai/).

### Ollama Cloud — REVIEW REQUIRED / NOT_VERIFIED

- Current `gpt-oss:120b` hosted configuration is not proven permanent-zero-cost.
- A Free account tier or starter allowance does not establish this model's
  non-billable eligibility. Do not rely on temporary credits or subscriptions.
- Registry disables hosted Ollama even when a key is configured; evaluation
  rejects its unverified model before any provider client/call.
- Local Ollama can have zero provider API fee but requires local hardware. The
  current hosted adapter is not reclassified or replaced with a local backend.
- Evidence: repository hosted endpoint/model and reviewed requirement to withhold
  eligibility; no starter-model or billing evidence was obtained.
- Revalidation: selected-model inclusion in a non-billable starter allowance,
  no required payment method and hard quota failure via
  [Ollama documentation](https://docs.ollama.com/).

### Kilo — KEEP WITH STRONG MODEL GUARD / CONDITIONAL_FREE

- Only `kilo-auto/free` or models with current evidence of both $0 input and
  $0 output qualify. Reviewed zero-price catalog evidence is not committed here.
- Current conservative guard recognizes only `kilo-auto/free`. The existing
  anonymous adapter passes model names verbatim and has no Authorization/billing
  setup; the alias is supported without a new adapter/routing implementation.
- Historical defaults `stepfun/step-3.7-flash:free`,
  `cohere/north-mini-code:free`, `dots-studio/dots-3-note-preview:free` are
  NOT_VERIFIED. Historical `:free` promotions are not durable pricing evidence.
- Defaults are preserved for audit history; enabling Kilo with any unverified
  configured task model leaves the provider disabled. No silent model replacement.
- To use the reviewed alias, explicitly set all three Kilo task model settings
  to `kilo-auto/free`, retaining `KILO_ENABLED=true` only after operator review.
- Free capacity exhaustion/disappearance must fail; never select a paid variant.
- NVIDIA stays excluded. No NVIDIA model is added or approved.
- Revalidation: current anonymous/free route availability and any model's
  input/output price evidence via [Kilo documentation](https://kilo.ai/docs).

### LLM7 — KEEP WITH FREE-QUOTA GUARD / CONDITIONAL_FREE

- Current allowed model profile: `gpt-oss:20b` for all three task settings, only
  after verifying it belongs to non-billable free usage for the operator account.
- Free token quota is acceptable only when its exhaustion fails rather than bills.
- Paid balance/subscription/automatic fall-through is prohibited.
- Registry requires existing opt-in/key plus `LLM7_ZERO_COST_VERIFIED=true`.
  Unknown models remain blocked; billing errors never enable payments or change
  the model. SDK retries/fallback stay within otherwise eligible free profiles.
- Evidence: reviewed free-quota product input and repository defaults; actual
  account/model/quota configuration was not verified remotely.
- Revalidation: exact model free inclusion and non-billable over-quota behavior
  via the [LLM7 provider portal](https://llm7.io/).

### OpenAI direct — REJECTED_PAID

- Decision: **REJECTED — ZERO-COST GATE FAILED**.
- Direct API paid usage/billing eligibility does not satisfy mandatory permanent
  zero-cost requirements. Trial/promotional credits do not change this decision.
- No allowed production or live evaluation model under the current policy.
- Registry never constructs it, including with a configured key. Primary/fallback
  exclusion remains; evaluation returns a controlled rejection before settings
  access/client construction. Live opt-ins cannot bypass the rejection.
- Responses adapter and its network-free contract/security tests are retained.
- Historical smoke: one attempt, FAILED / rate_limit. That error proves neither
  adapter failure nor pricing; rejection is the reviewed business-policy decision.
- Revalidation requires an explicitly reviewed permanent no-payment/no-overage
  offering and policy change, not another smoke or credit purchase. Official
  revalidation location: [OpenAI platform](https://platform.openai.com/docs).

## Phase 5 candidate decisions

Preferred roadmap evaluation order remains unchanged; gate outcomes determine
which candidate can proceed to onboarding.

| Candidate | Decision | Reason / next evidence |
|---|---|---|
| 1. OpenAI direct | REJECTED: ZERO-COST GATE | Paid direct API; retained tested adapter, no live use |
| 2. DeepSeek direct | REJECTED BEFORE ADAPTER: ZERO-COST GATE | Token-priced API |
| 3. Mistral direct | NEXT ELIGIBLE CANDIDATE | Reviewed Free mode, API keys, no required credit card, included monthly usage, pay-as-you-go can stay disabled |
| 4. Anthropic direct | REJECTED BEFORE ADAPTER: ZERO-COST GATE | Paid API; trial/promotional credits insufficient |
| 5. xAI direct | REJECTED BEFORE ADAPTER: ZERO-COST GATE | Prepaid/invoiced paid API |

Mistral is not admitted and no adapter is implemented in this audit. Before
onboarding, verify the account is Free mode, no card is needed, pay-as-you-go is
disabled, the selected model is included and over-quota behavior fails closed.
Record repeatable evidence; provider-level Free mode does not admit every model.

## Fail-closed rules

When free quota ends, a free model disappears or eligibility cannot be proven:

1. Return the existing safe provider/configuration failure.
2. Fallback only to another eligible provider/model in a verified free mode.
3. Never switch to a paid variant, remove a free suffix, upgrade an account,
   activate billing/overage, buy credits or start a subscription.
4. Require review of changed evidence before re-enabling use.

Registry guards remove unsafe configurations before production routing can call
them. Task-chain priority/order, request/SSE contracts and routing/fallback
algorithms are preserved. Unknown optional profiles can reduce available fallback
capacity; that is intentional fail-closed behavior, not a paid fallback. No Phase
6 policy engine or Phase 7 adaptive routing is introduced.
