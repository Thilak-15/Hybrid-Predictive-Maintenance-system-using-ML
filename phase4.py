import numpy as np
import matplotlib.pyplot as plt
import joblib

# ------------ Load Models & Data --------------

# NOTE: Make sure these files exist in the directory!
# Models trained in earlier phases (hopefully still compatible)
clf_model = joblib.load('classification_model.joblib')    # Classification model from Phase 2
reg_model = joblib.load('regression_model.joblib')        # Regression model from Phase 3

# Test datasets from previous phases
# I think Phase 2 and 3 used different splits, so need to be careful here
X_test_cls = joblib.load('X_test.joblib')       # Classification set
X_test_reg = joblib.load('X_test_reg.joblib')   # Regression set

# Just using classification test set for simplicity now
X_test = X_test_cls    # NOTE: Inconsistent use, might need syncing later

# ------------ Compute Risk Components ------------

# 1. Probabilities from classifier
# This should give us 5 class probs per sample, class 4 = failure
probabilities = clf_model.predict_proba(X_test)
# Extract failure probability only
try:
    fail_prob = probabilities[:, 4]
except IndexError:
    print("Oops - classifier might not be trained for 5 classes.")
    fail_prob = np.zeros(X_test.shape[0])

# 2. Predicted remaining time from regression model
# We'll prevent weird negative predictions just in case
try:
    est_time_left = reg_model.predict(X_test)
except Exception as e:
    print("Something went wrong with regression prediction:", e)
    est_time_left = np.ones(X_test.shape[0]) * 100  # fallback guess

# Make sure no zero or negative times
est_time_left = np.clip(est_time_left, 1e-6, None)

# 3. Risk = Prob * Time (this is just one way to define it)
risk_values = fail_prob * est_time_left

# 4. Normalize to [0,1] (avoid divide by zero just in case)
risk_min = risk_values.min()
risk_max = risk_values.max()
denom = (risk_max - risk_min) if (risk_max - risk_min) != 0 else 1e-12
normalized_risk = (risk_values - risk_min) / denom

# 5. Urgency = Prob / Time (this emphasizes short remaining life)
urgency_risk = fail_prob / (est_time_left + 1e-6)   # avoids division by 0

# 6. Threshold-based alerting
risk_threshold = 0.7
is_alert = normalized_risk > risk_threshold

# Quick summary
print(f"ALERTS: {is_alert.sum()} out of {len(is_alert)} samples triggered maintenance warning.")

# ----------- Visuals ------------------

# Normalized risk plot
plt.figure(figsize=(12, 6))
plt.plot(normalized_risk, color='steelblue', label='Normalized Risk')
plt.axhline(risk_threshold, color='red', linestyle='--', label=f'Threshold ({risk_threshold})')
plt.title('Normalized Risk Score')
plt.xlabel('Sample Index')
plt.ylabel('Risk Score (0-1)')
plt.legend()
plt.tight_layout()
plt.show()

# Urgency-based score plot
plt.figure(figsize=(12, 6))
plt.plot(urgency_risk, label='Urgency Score', color='darkorange')
plt.title('Urgency-Based Risk Score')
plt.xlabel('Sample Index')
plt.ylabel('Urgency')
plt.legend()
plt.tight_layout()
plt.show()

# ----------- Individual Alerts --------------

# Print out samples that triggered alerts
for idx in range(len(is_alert)):
    if is_alert[idx]:
        # Technically redundant formatting but clearer this way
        score = normalized_risk[idx]
        print(f"[Alert] Sample #{idx} => Risk Score: {score:.3f}")