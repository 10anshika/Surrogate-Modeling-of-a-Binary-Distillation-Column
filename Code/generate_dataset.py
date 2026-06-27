"""
Final dataset — uses direct VLE-based column model with proper variance
"""
import numpy as np
import pandas as pd
np.random.seed(42)
N = 500

def alpha_BT(T_K):
    """Relative volatility B-T from Antoine equations"""
    Ps_b = 10**(6.90565 - 1211.033/(220.790 + T_K - 273.15))
    Ps_t = 10**(6.95464 - 1344.800/(219.482 + T_K - 273.15))
    return Ps_b / Ps_t

records = []
for _ in range(N):
    z_F    = np.random.uniform(0.25, 0.75)
    P_col  = np.random.uniform(85, 115)
    T_feed = np.random.uniform(320, 380)
    q      = np.random.uniform(0.0, 1.2)          # feed quality
    N_stg  = int(np.random.randint(8, 35))
    N_feed = int(np.random.randint(3, max(4, N_stg-2)))
    RR     = np.random.uniform(1.2, 7.0)
    D_rate = np.random.uniform(20, 60)             # distillate rate kmol/hr
    F_rate = 100.0
    B_rate = F_rate - D_rate

    # Approx column temp
    T_col  = 363 - 15*(P_col - 101.3)/101.3
    alpha  = alpha_BT(T_col)
    alpha  = np.clip(alpha, 1.05, 4.5)

    # Fenske Nmin
    x_D_t = 0.98; x_B_t = 0.02
    N_min = np.log((x_D_t/(1-x_D_t))*((1-x_B_t)/x_B_t)) / np.log(alpha)
    N_min = max(N_min, 2.0)

    # RR_min (Underwood, simplified)
    RR_min = (alpha/(alpha-1)) * (x_D_t - z_F) / (x_D_t - z_F * alpha/(alpha-1+1e-9))
    RR_min = max(abs(RR_min), 0.6)

    # Actual performance — separation efficiency
    stage_ratio = N_stg / N_min        # >1 is over-designed
    rr_ratio    = RR / RR_min          # >1 is above minimum
    eff = 1 - np.exp(-0.6*(stage_ratio - 1) - 0.4*(rr_ratio - 1))
    eff = np.clip(eff, 0.01, 0.97)

    # Actual distillate & bottoms purity (from overall material balance + efficiency)
    # Total benzene in feed = z_F * F = x_D*D + x_B*B
    # With efficiency, approach towards ideal separation
    x_D = x_D_t * eff + z_F * (1 - eff) * (1 + (RR-RR_min)*0.02)
    x_B_max = (z_F * F_rate - x_D * D_rate) / B_rate
    x_B = max(0.001, x_B_max * (1 - eff*0.8))

    # Clamp physically
    x_D = np.clip(x_D + np.random.normal(0, 0.004), 0.50, 0.999)
    x_B = np.clip(x_B + np.random.normal(0, 0.002), 0.001, 0.35)

    # Energy balance
    # Latent heats (kJ/mol)
    lam_mix_D = 30.7*x_D + 33.2*(1-x_D)
    lam_mix_B = 30.7*x_B + 33.2*(1-x_B)
    Q_C = (RR + 1) * D_rate * lam_mix_D            # kJ/hr → kW (* 1/3.6)
    Q_C /= 3.6
    # Feed enthalpy contribution
    dH_feed = F_rate * q * lam_mix_D / 3.6
    Q_R = Q_C - dH_feed + B_rate * lam_mix_B / 3.6

    Q_C = abs(Q_C) + np.random.normal(0, 15)
    Q_R = abs(Q_R) + np.random.normal(0, 15)

    records.append({
        'feed_temperature_K':    round(T_feed, 2),
        'feed_pressure_kPa':     round(P_col, 2),
        'feed_composition_benz': round(z_F, 4),
        'num_stages':            N_stg,
        'feed_stage':            N_feed,
        'reflux_ratio':          round(RR, 3),
        'distillate_rate_kmolhr':round(D_rate, 2),
        'bottoms_rate_kmolhr':   round(B_rate, 2),
        'feed_quality_q':        round(q, 4),
        'column_pressure_kPa':   round(P_col, 2),
        'rel_volatility':        round(alpha, 4),
        'distillate_purity_xD':  round(x_D, 4),
        'bottoms_purity_xB':     round(x_B, 4),
        'condenser_duty_kW':     round(Q_C, 2),
        'reboiler_duty_kW':      round(Q_R, 2),
    })

df = pd.DataFrame(records)
df.to_csv('/home/claude/Dataset.csv', index=False)
print("Shape:", df.shape)
print(df[['distillate_purity_xD','bottoms_purity_xB','condenser_duty_kW','reboiler_duty_kW']].describe().round(4))
