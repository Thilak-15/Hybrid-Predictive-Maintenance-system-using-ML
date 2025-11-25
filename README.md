# Predictive Maintenance Project (Simplified)

## Group Contribution

Hemanth Varma Kanumuri (SE23UARI062): 30%
Varshita Kinjarapu (SE23UCSE181):25%
Rihanth Katkuri (SE23UARI062): 15%
Vishwatej Goud (SE23UARI072): 15%
Thilak Kusampudi (SE23UCSE099): 15%


## What's This About?

This project helps fix machines before they break. Instead of just saying “something will go wrong,” we wanted to figure out how soon it might happen. We used real sensor data from engines to build a tool that shows how much wear-and-tear there is, predicts how much longer the machine will last, and helps decide the best time to do maintenance.

## Why We Did It

Most systems just give you a yes/no answer — “Will it break?” That’s helpful, but not enough. We wanted to give teams a better idea of how serious a problem is and how soon it might happen. By breaking down the machine’s condition into several stages, people can plan better and avoid wasting time or money on unnecessary repairs.

## The Data We Used

We used the CMAPSS dataset, which is like a simulation of real jet engine data. It includes readings from many different sensors over time.

* Each engine gives off data for every cycle (like time steps).
* We paid special attention to Engine ID, the cycle count, and 24 sensor readings.

## What We Built

### Step 1: Sorting Out Engine Health Stages

We used a method called KMeans to group the engine’s condition into five stages:

* Stage 0: All good
* Stage 1: Slight wear
* Stage 2: Medium wear
* Stage 3: Serious condition
* Stage 4: About to fail

We also used some math tricks to turn the data into simpler visuals, so it’s easier to understand.

### Step 2: Figuring Out the Current Stage

We trained a model (logistic regression) to guess which stage the engine is in, just by looking at the sensor data.

* Since most engines were in early stages, we had to adjust for that so the model didn’t get biased.
* We checked how well the model worked using scores like precision, recall, and a confusion matrix.
* We also looked at which sensors were the most helpful for the model.

### Step 3: Estimating Time Left

We built another model (this time, a random forest regressor) to guess how many cycles are left before the engine gets worse.

* We labeled the data by counting how long it took to move from one stage to the next.
* We used common scoring tools like RMSE, MAE, and R² to see how close our guesses were.
* We also plotted some graphs to compare predictions with real outcomes.

### Step 4: Risk Score and Alerts

This is where it all comes together:

* The classifier tells us how bad the situation is right now.
* The regressor tells us how much time we have left.
* We combined these to create a risk score — kind of like a warning level.
* We scaled the score between 0 and 1, and if it goes above a set point, we trigger an alert.
* We also made graphs to show how the risk changes over time.

## Project Files

Here’s what’s in the project folder:

* phase2.py – Code for classifying the current stage.
* phase3.py – Code for predicting time left.
* phase4.py – Code for creating the risk score and alerts.
* classification_model.joblib – Saved model for classification.
* regression_model.joblib – Saved model for regression.
* X_test.joblib, X_test_reg.joblib – Test data.
* README.md – This doc!

## A Few Things to Know

* The test data isn’t exactly the same in every phase, but Phase 4 uses the classification test set to keep it consistent.
* You can tweak the alert level depending on whether you care more about catching all issues or avoiding false alarms.

## License

Feel free to use or change this project — it’s under the MIT License.
