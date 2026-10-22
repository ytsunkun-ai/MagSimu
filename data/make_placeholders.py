from pathlib import Path
import numpy as np
import pandas as pd

raw = Path("data/raw"); raw.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261023)
MU0=4e-7*np.pi

# Synthetic B-H loop, explicitly labelled.  It is generated from a known
# reduced-order model only to exercise the fitting code.
H_up = np.concatenate([np.linspace(-4e5,-2e4,60), np.linspace(-2e4,2e4,160), np.linspace(2e4,4e5,60)])
H_down = H_up[::-1]
Ms=1.22e6; Hc=110.0; a=5200.0; phi=0.60
B_up = MU0*(H_up+phi*Ms*np.tanh((H_up-Hc)/a))
B_down = MU0*(H_down+phi*Ms*np.tanh((H_down+Hc)/a))
B_up += rng.normal(0,0.004,B_up.size); B_down += rng.normal(0,0.004,B_down.size)
bh = pd.DataFrame({
    "H_A_m":np.concatenate([H_up,H_down]),
    "B_T":np.concatenate([B_up,B_down]),
    "branch":["ascending"]*H_up.size+["descending"]*H_down.size,
    "frequency_hz":100000.0,
    "source":"synthetic_placeholder_generated_for_code_verification",
})
bh.to_csv(raw/"example_bh_placeholder.csv",index=False)

rows=[]
for f in [5e4,1e5,2e5,4e5]:
    for B in [0.02,0.05,0.10]:
        P=1.2e-4*(f**1.15)*(B**1.75)*(1+rng.normal(0,0.03))
        P=max(P,1e-9)
        total=P*7150.0
        rows.append({"frequency_hz":f,"B_peak_T":B,"loss_W_kg":P,
                     "loss_total_W_m3":total,
                     "loss_hysteresis_W_m3":0.55*total,
                     "loss_eddy_W_m3":0.30*total,
                     "loss_excess_W_m3":0.15*total,
                     "temperature_K":293.15,"source":"synthetic_placeholder_generated_for_code_verification"})
pd.DataFrame(rows).to_csv(raw/"example_loss_placeholder.csv",index=False)

n=48
shape=7.0
scale=4.5e7
fields=scale*(-np.log(np.clip(rng.random(n),1e-9,1-1e-9)))**(1/shape)
thickness=rng.lognormal(np.log(1.0e-7)-0.5*0.2**2,0.2,n)
V=fields*thickness
pd.DataFrame({"breakdown_voltage_V":V,"thickness_m":thickness,"breakdown_field_V_m":fields,
              "source":"synthetic_placeholder_generated_for_code_verification"}).to_csv(raw/"example_breakdown_placeholder.csv",index=False)
print("wrote", [str(p) for p in sorted(raw.glob('*.csv'))])
