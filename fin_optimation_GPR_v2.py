import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution, NonlinearConstraint


# 1. Load and clean data
data = pd.read_excel("C:/Users/domin/OneDrive/Desktop/IREC/Fin Optimization (1).xlsx", header=1) #read in the data
data = data.dropna(subset=['Factor of Safety', 'Weight (lbs)', 'Center of Pressure (in)']) #drop the blanks

X = data[['Root Chord', 'Tip Chord', 'Semi-span']] #training data inputs
y_fos = data['Factor of Safety'] #training data outputs
y_cp = data['Center of Pressure (in)']


# 2. Normalize features and targets (We normalize so that it treats all 3 scales with equal importance)
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)

scaler_y_fos = StandardScaler().fit(y_fos.values.reshape(-1, 1))
y_fos_scaled = scaler_y_fos.transform(y_fos.values.reshape(-1, 1)).ravel()

scaler_y_cp = StandardScaler().fit(y_cp.values.reshape(-1, 1))
y_cp_scaled = scaler_y_cp.transform(y_cp.values.reshape(-1, 1)).ravel()


# 3. GPR Models for FoS and CP
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0,1.0,1.0], nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2)) #fixed noise since we trust the inputs

gp_fos = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp_cp  = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

gp_fos.fit(X_scaled, y_fos_scaled)
gp_cp.fit(X_scaled, y_cp_scaled)

""" This is to visualize prediction accuracy.  Mostly to see if it is overfitting or underfitting
import matplotlib.pyplot as plt
from itertools import product

# Scale the training features
X_scaled_for_plot = scaler_X.transform(X)

# Predict CP on training data
cp_pred_scaled, cp_std_scaled = gp_cp.predict(X_scaled_for_plot, return_std=True)

# Convert predictions back to original units
cp_pred = scaler_y_cp.inverse_transform(cp_pred_scaled.reshape(-1,1)).ravel()

# Plot predicted vs true CP
plt.figure(figsize=(6,6))
plt.scatter(y_cp, cp_pred, alpha=0.5)
plt.xlabel("True CP")
plt.ylabel("Predicted CP")
plt.plot([105,107],[105,107],'r--')
plt.title("GPR CP Predictions vs True Values")
plt.show()
"""


# 4. Deterministic weight calculation (Calculates the weight for us)
def compute_weight(root_chord, tip_chord, semi_span):
    area1 = tip_chord * semi_span
    area2 = (root_chord - tip_chord) * semi_span / 2
    total_area = area1 + area2

    thickness = 0.1875
    density   = 0.0975436

    volume = total_area * thickness
    fin_weight = volume * density
    total_weight = fin_weight * 4

    return total_weight


# 5. Objective function
FOS_MIN = 1.15 #Factor of Safety we want
CONFIDENCE_MULTIPLIER = 1.96 #~95% confidence interval

def objective(x):
    R, T, S = x
    df = pd.DataFrame([[R, T, S]], columns=['Root Chord', 'Tip Chord', 'Semi-span'])
    df_scaled = scaler_X.transform(df)

    fos_mean_scaled, fos_std_scaled = gp_fos.predict(df_scaled, return_std=True)
    fos_mean = float(scaler_y_fos.inverse_transform([[fos_mean_scaled[0]]])[0,0])
    fos_std  = float(scaler_y_fos.scale_[0] * fos_std_scaled[0])
    fos_lcb  = fos_mean - CONFIDENCE_MULTIPLIER * fos_std

    weight = compute_weight(R, T, S)

    penalty = 0
    if fos_lcb <= FOS_MIN:
        penalty += 1000 * (FOS_MIN - fos_lcb)

    return float(weight + penalty)


# 6. CP constraints
def cp_lower_constraint(x):
    R, T, S = x
    df = pd.DataFrame([[R, T, S]], columns=['Root Chord', 'Tip Chord', 'Semi-span'])
    df_scaled = scaler_X.transform(df)
    cp_mean_scaled, _ = gp_cp.predict(df_scaled, return_std=True)
    cp_mean = float(scaler_y_cp.inverse_transform([[cp_mean_scaled[0]]])[0,0])
    return cp_mean

def cp_upper_constraint(x):
    R, T, S = x
    df = pd.DataFrame([[R, T, S]], columns=['Root Chord', 'Tip Chord', 'Semi-span'])
    df_scaled = scaler_X.transform(df)
    cp_mean_scaled, _ = gp_cp.predict(df_scaled, return_std=True)
    cp_mean = float(scaler_y_cp.inverse_transform([[cp_mean_scaled[0]]])[0,0])
    return cp_mean

nlc_cp_low = NonlinearConstraint(cp_lower_constraint, 90.8, np.inf)
nlc_cp_high = NonlinearConstraint(cp_upper_constraint, -np.inf, 91.2)



# 7. Differential evolution optimization
bounds = [
    (10, 20),  # Root chord bounds
    (3, 6),    # Tip chord bounds
    (4.9, 6)     # Semi-span bounds
]

result = differential_evolution(
    objective,
    bounds,
    constraints=(nlc_cp_low, nlc_cp_high),
    strategy='best1bin',
    maxiter=800,
    popsize=15,
    seed=42,
    tol=1e-7,
    polish=True,
)



# 8. Extract optimal result
R_opt, T_opt, S_opt = result.x
df_opt = pd.DataFrame([[R_opt, T_opt, S_opt]], columns=['Root Chord', 'Tip Chord', 'Semi-span'])
df_opt_scaled = scaler_X.transform(df_opt)

fos_mean_scaled, fos_std_scaled = gp_fos.predict(df_opt_scaled, return_std=True)
fos_mean = float(scaler_y_fos.inverse_transform([[fos_mean_scaled[0]]])[0,0])
fos_std  = float(scaler_y_fos.scale_[0] * fos_std_scaled[0])
fos_lcb  = fos_mean - CONFIDENCE_MULTIPLIER * fos_std

cp_mean_scaled, _ = gp_cp.predict(df_opt_scaled, return_std=True)
cp_mean = float(scaler_y_cp.inverse_transform([[cp_mean_scaled[0]]])[0,0])

weight = compute_weight(R_opt, T_opt, S_opt)


# 9. Print results
print("\n================ OPTIMAL FIN DESIGN =================\n")
print(f"Root Chord: {R_opt:.3f} in")
print(f"Tip Chord:  {T_opt:.3f} in")
print(f"Semi-span:  {S_opt:.3f} in")

print("\n---- Performance Predictions ----")
print(f"Predicted FOS: {fos_mean:.3f} ± {fos_std:.3f}")
print(f"Conservative FOS (95% LCB): {fos_lcb:.3f}")
print(f"Predicted CP: {cp_mean:.3f}")
print(f"Calculated Weight (4 fins): {weight:.3f} lbs")
print("\n======================================================\n")
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import differential_evolution


############################################################
# 1. TRAIN/TEST VALIDATION
############################################################

def validate_train_test_split(X, y, gp, scaler_X, scaler_y, label="FoS"):
    print(f"\n===== TRAIN/TEST VALIDATION for {label} =====")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.transform(y_train.values.reshape(-1,1)).ravel()

    # Fit a NEW model identical to your original
    gp.fit(X_train_scaled, y_train_scaled)

    # Predict test set
    y_pred_scaled, std_scaled = gp.predict(X_test_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    std = scaler_y.scale_[0] * std_scaled

    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # Confidence interval coverage
    lower_95 = y_pred - 1.96 * std
    coverage = np.mean(y_test >= lower_95)
    print(f"95% CI Coverage = {coverage*100:.1f}%")

    # Plot predictions
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    minv, maxv = min(y_test), max(y_test)
    plt.plot([minv,maxv],[minv,maxv],'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{label}: Prediction vs True")
    plt.grid(True)
    plt.show()

    # Residuals
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_test - y_pred, alpha=0.6)
    plt.axhline(0, color='r')
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.title(f"{label}: Residual Plot")
    plt.grid(True)
    plt.show()


############################################################
# 2. CROSS VALIDATION
############################################################

def cross_validation_scores(X, y, gp, scaler_X, scaler_y, label="FoS"):
    print(f"\n===== CROSS VALIDATION for {label} =====")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    coverages = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.transform(y_train.values.reshape(-1,1)).ravel()

        # Fit
        gp.fit(X_train_scaled, y_train_scaled)

        # Predict
        pred_scaled, std_scaled = gp.predict(X_test_scaled, return_std=True)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
        std = scaler_y.scale_[0] * std_scaled

        rmse = mean_squared_error(y_test, pred, squared=False)
        rmses.append(rmse)

        lower_95 = pred - 1.96*std
        coverages.append(np.mean(y_test >= lower_95))

    print(f"Average RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"95% CI Coverage: {np.mean(coverages)*100:.1f}%")



############################################################
# 3. KERNEL + HYPERPARAMETER DIAGNOSTICS
############################################################

def inspect_gpr_kernel(gp, label="FoS"):
    print(f"\n===== GPR KERNEL DIAGNOSTICS for {label} =====")
    print(gp.kernel_)



############################################################
# 4. SENSITIVITY SWEEPS
############################################################
"""
def sensitivity_sweep(gp, scaler_X, scaler_y, label="FoS"):
    print(f"\n===== SENSITIVITY ANALYSIS for {label} =====")

    root_vals = np.linspace(14, 20, 30)
    tip_vals  = np.linspace(3, 6, 30)
    span_vals = np.linspace(4, 6, 30)

    def sweep(var_name, values):
        preds = []
        for v in values:
            test = pd.DataFrame([{
                "Root Chord": 16,
                "Tip Chord": 4.5,
                "Semi-span": 5,
                var_name: v
            }])

            Xs = scaler_X.transform(test)
            mean_s, _ = gp.predict(Xs, return_std=True)
            preds.append(scaler_y.inverse_transform(mean_s.reshape(-1,1))[0,0])
        return preds

    plt.figure(figsize=(8,6))
    plt.plot(root_vals, sweep("Root Chord", root_vals), label="Root chord")
    plt.plot(tip_vals,  sweep("Tip Chord", tip_vals), label="Tip chord")
    plt.plot(span_vals, sweep("Semi-span", span_vals), label="Semi-span")

    plt.xlabel("Parameter Value")
    plt.ylabel(f"{label} Prediction")
    plt.title(f"Sensitivity Sweep — {label}")
    plt.legend()
    plt.grid(True)
    plt.show()


"""
############################################################
# 5. OPTIMIZER STABILITY TEST
############################################################

def test_optimizer_stability(objective, bounds, constraints):
    print("\n===== OPTIMIZER STABILITY TEST =====")

    results = []
    for seed in range(6):
        print(f"Running DE with seed {seed}...")
        res = differential_evolution(
            objective, bounds, constraints=constraints,
            seed=seed, maxiter=200, popsize=10
        )
        results.append(res.x)

    results = np.array(results)
    print("\nSolutions:")
    print(results)
    print("\nStd deviation across runs (lower is better):")
    print(results.std(axis=0))



############################################################
# 6. MASTER FUNCTION TO RUN EVERYTHING
############################################################

def run_full_validation(X, y_fos, y_cp, gp_fos, gp_cp,
                        scaler_X, scaler_y_fos, scaler_y_cp,
                        objective=None, bounds=None, constraints=None):

    validate_train_test_split(X, y_fos, gp_fos, scaler_X, scaler_y_fos, label="FOS")
    validate_train_test_split(X, y_cp,  gp_cp,  scaler_X, scaler_y_cp,  label="CP")

    cross_validation_scores(X, y_fos, gp_fos, scaler_X, scaler_y_fos, label="FOS")
    cross_validation_scores(X, y_cp,  gp_cp,  scaler_X, scaler_y_cp,  label="CP")

    inspect_gpr_kernel(gp_fos, label="FOS")
    inspect_gpr_kernel(gp_cp,  label="CP")

#    sensitivity_sweep(gp_fos, scaler_X, scaler_y_fos, label="FOS")
#    sensitivity_sweep(gp_cp,  scaler_X, scaler_y_cp,  label="CP")

    if objective is not None:
        test_optimizer_stability(objective, bounds, constraints)

    print("\n===== VALIDATION COMPLETE =====\n")
run_full_validation(
    X, y_fos, y_cp,
    gp_fos, gp_cp,
    scaler_X, scaler_y_fos, scaler_y_cp,
    objective=objective,
    bounds=bounds,
    constraints=(nlc_cp_low, nlc_cp_high)
)
#%%
'''
Root=10
Tip=3.25
Span=4.9
weight = compute_weight(Root, Tip, Span)
print(weight)
'''

