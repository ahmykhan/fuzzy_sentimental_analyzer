import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Define Antecedents (Inputs) and Consequent (Output)
# Input 1: Positivity score (0 to 1)
pos_score = ctrl.Antecedent(np.linspace(0, 1, 100), 'pos_score')
# Input 2: Negativity score (0 to 1)
neg_score = ctrl.Antecedent(np.linspace(0, 1, 100), 'neg_score')
# Output: Final sentiment rating (-1 to 1)
sentiment = ctrl.Consequent(np.linspace(-1, 1, 100), 'sentiment')

# 2. Define Membership Functions (Low, Medium, High)
pos_score['low'] = fuzz.trimf(pos_score.universe, [0.0, 0.0, 0.5])
pos_score['high'] = fuzz.trimf(pos_score.universe, [0.5, 1.0, 1.0])

neg_score['low'] = fuzz.trimf(neg_score.universe, [0.0, 0.0, 0.5])
neg_score['high'] = fuzz.trimf(neg_score.universe, [0.5, 1.0, 1.0])

sentiment['negative'] = fuzz.trimf(sentiment.universe, [-1.0, -1.0, 0.0])
sentiment['neutral']  = fuzz.trimf(sentiment.universe, [-0.5, 0.0, 0.5])
sentiment['positive'] = fuzz.trimf(sentiment.universe, [0.0, 1.0, 1.0])

# 3. Define 3 Simple Fuzzy Rules (as required by Task 1.5)
rule1 = ctrl.Rule(pos_score['high'] & neg_score['low'], sentiment['positive'])
rule2 = ctrl.Rule(pos_score['low'] & neg_score['high'], sentiment['negative'])
rule3 = ctrl.Rule(pos_score['low'] & neg_score['low'], sentiment['neutral'])

# 4. Build Control System and Simulator
sentiment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
sentiment_sim = ctrl.ControlSystemSimulation(sentiment_ctrl)

# 5. Test with sample values
sentiment_sim.input['pos_score'] = 0.8
sentiment_sim.input['neg_score'] = 0.1
sentiment_sim.compute()

print(f"Test Input: Pos=0.8, Neg=0.1")
print(f"Fuzzy Output Score: {sentiment_sim.output['sentiment']:.3f}")

# 6. Plot the membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

pos_score.view(ax=ax0)
ax0.set_title("Positive Score Membership")

neg_score.view(ax=ax1)
ax1.set_title("Negative Score Membership")

sentiment.view(ax=ax2)
ax2.set_title("Sentiment Output Membership")

plt.tight_layout()
plt.savefig("practice/fuzzy_plot.png")
print("Saved membership plot to practice/fuzzy_plot.png")
plt.show()