import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, shap, warnings, json, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

os.makedirs('/home/claude/plots', exist_ok=True)

df = pd.read_csv('/home/claude/Dataset.csv')
FEATURES = ['feed_temperature_K','feed_pressure_kPa','feed_composition_benz',
            'num_stages','feed_stage','reflux_ratio','distillate_rate_kmolhr',
            'bottoms_rate_kmolhr','feed_quality_q','rel_volatility']
TARGETS  = ['distillate_purity_xD','bottoms_purity_xB','condenser_duty_kW','reboiler_duty_kW']
TLABELS  = ['Distillate Purity (xD)','Bottoms Purity (xB)','Condenser Duty QC (kW)','Reboiler Duty QR (kW)']

X = df[FEATURES].values; y = df[TARGETS].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

models = {
    'Linear Regression': (MultiOutputRegressor(LinearRegression()), True),
    'Random Forest':     (RandomForestRegressor(200, max_depth=15, random_state=42, n_jobs=-1), False),
    'XGBoost':           (MultiOutputRegressor(xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)), False),
    'ANN (MLP)':         (MLPRegressor(hidden_layer_sizes=(256,128,64), activation='relu', max_iter=2000,
                           learning_rate_init=0.001, random_state=42,
                           early_stopping=True, validation_fraction=0.1), True),
}

results = {}; preds = {}
for name, (m, scaled) in models.items():
    Xtr = X_tr_s if scaled else X_tr
    Xte = X_te_s if scaled else X_te
    m.fit(Xtr, y_tr)
    yp = m.predict(Xte)
    preds[name] = yp
    mets = {}
    for i, t in enumerate(TARGETS):
        mets[t] = {'MAE': float(mean_absolute_error(y_te[:,i], yp[:,i])),
                   'RMSE': float(np.sqrt(mean_squared_error(y_te[:,i], yp[:,i]))),
                   'R2': float(r2_score(y_te[:,i], yp[:,i]))}
    results[name] = mets
    avg = np.mean([mets[t]['R2'] for t in TARGETS])
    print(f"{name:22s} | Avg R²={avg:.4f}")

best = max(results, key=lambda m: np.mean([results[m][t]['R2'] for t in TARGETS]))
print(f"\nBest: {best}")

COLORS = ['#2E86AB','#A23B72','#F18F01','#C73E1D']

# Fig 1 — Pred vs Actual (best model)
yp_b = preds[best]
fig, axes = plt.subplots(2,2,figsize=(11,9))
fig.suptitle(f'Predicted vs Actual — {best}', fontsize=13, fontweight='bold')
for i,(ax,t,lb,c) in enumerate(zip(axes.flat,TARGETS,TLABELS,COLORS)):
    ax.scatter(y_te[:,i], yp_b[:,i], alpha=0.45, s=18, color=c)
    mn,mx = y_te[:,i].min(), y_te[:,i].max()
    ax.plot([mn,mx],[mn,mx],'k--',lw=1.2)
    r2=results[best][t]['R2']; rmse=results[best][t]['RMSE']
    ax.set_title(f'{lb}\nR²={r2:.4f}  RMSE={rmse:.4f}',fontsize=10)
    ax.set_xlabel('Actual',fontsize=9); ax.set_ylabel('Predicted',fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/plots/fig1_pred_vs_actual.png',dpi=150,bbox_inches='tight'); plt.close()

# Fig 2 — R² heatmap
mnames = list(results.keys())
r2mat = np.array([[results[m][t]['R2'] for t in TARGETS] for m in mnames])
fig,ax = plt.subplots(figsize=(9,4))
im = ax.imshow(r2mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im,ax=ax,label='R²')
ax.set_xticks(range(4)); ax.set_xticklabels(['xD','xB','QC','QR'],fontsize=11)
ax.set_yticks(range(4)); ax.set_yticklabels(mnames,fontsize=11)
ax.set_title('R² Heatmap: All Models × All Targets',fontsize=12,fontweight='bold')
for i in range(4):
    for j in range(4):
        ax.text(j,i,f'{r2mat[i,j]:.3f}',ha='center',va='center',fontsize=10,fontweight='bold',
                color='black' if r2mat[i,j]>0.5 else 'white')
plt.tight_layout()
plt.savefig('/home/claude/plots/fig2_r2_heatmap.png',dpi=150,bbox_inches='tight'); plt.close()

# Fig 3 — Bar comparison (MAE, RMSE, R2 averaged)
fig,axes = plt.subplots(1,3,figsize=(14,5))
fig.suptitle('Average Model Metrics (across all 4 targets)',fontsize=12,fontweight='bold')
bc=['#264653','#2A9D8F','#E9C46A','#E76F51']
for mi,metric in enumerate(['MAE','RMSE','R2']):
    ax=axes[mi]
    vals=[np.mean([results[mn][t][metric] for t in TARGETS]) for mn in mnames]
    bars=ax.bar(mnames,vals,color=bc,edgecolor='white')
    ax.set_title(f'Avg {metric}',fontsize=11,fontweight='bold')
    ax.set_xticklabels(mnames,rotation=20,ha='right',fontsize=9)
    ax.grid(axis='y',alpha=0.3)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+abs(b.get_height())*0.01,
                f'{v:.4f}',ha='center',va='bottom',fontsize=8)
plt.tight_layout()
plt.savefig('/home/claude/plots/fig3_model_comparison.png',dpi=150,bbox_inches='tight'); plt.close()

# Fig 4 — Feature importance (RF)
rf = [m for n,(m,s) in models.items() if n=='Random Forest'][0]
fig,axes = plt.subplots(2,2,figsize=(13,9))
fig.suptitle('Feature Importance — Random Forest',fontsize=13,fontweight='bold')
feat_labels=[f.replace('_','\n') for f in FEATURES]
for i,(ax,t,lb) in enumerate(zip(axes.flat,TARGETS,TLABELS)):
    imp=rf.estimators_[i].feature_importances_
    sidx=np.argsort(imp)
    ax.barh([feat_labels[j] for j in sidx],imp[sidx],color=plt.cm.Blues(np.linspace(0.3,0.9,len(FEATURES))))
    ax.set_title(lb,fontsize=10,fontweight='bold')
    ax.set_xlabel('Importance',fontsize=8)
    ax.tick_params(axis='y',labelsize=7)
    ax.grid(axis='x',alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/plots/fig4_feature_importance.png',dpi=150,bbox_inches='tight'); plt.close()

# Fig 5 — SHAP (XGBoost, xD)
xgb_model = [m for n,(m,s) in models.items() if n=='XGBoost'][0]
explainer = shap.TreeExplainer(xgb_model.estimators_[0])
sv = explainer.shap_values(X_te)
fig,ax = plt.subplots(figsize=(9,5))
shap.summary_plot(sv, X_te, feature_names=FEATURES, show=False, plot_type='bar')
plt.title('SHAP Values — XGBoost (Distillate Purity xD)',fontsize=11,fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/plots/fig5_shap.png',dpi=150,bbox_inches='tight'); plt.close()

# Fig 6 — Trend plots
rf_m = rf
base = np.array([350, 101.3, 0.5, 20, 10, 3.0, 40.0, 60.0, 0.5, 2.4])
rr_vals = np.linspace(1.5, 7.0, 60)
X_t1 = np.tile(base,(60,1)); X_t1[:,5]=rr_vals
xD_t = rf_m.predict(X_t1)[:,0]; xB_t = rf_m.predict(X_t1)[:,1]

fig,axes = plt.subplots(1,2,figsize=(12,4))
fig.suptitle('Physical Trend Validation',fontsize=12,fontweight='bold')
axes[0].plot(rr_vals,xD_t,'#2E86AB',lw=2.2,label='xD (distillate)')
axes[0].plot(rr_vals,xB_t,'#C73E1D',lw=2.2,ls='--',label='xB (bottoms)')
axes[0].set_xlabel('Reflux Ratio'); axes[0].set_ylabel('Mole Fraction')
axes[0].set_title('Purity vs Reflux Ratio\n(Expected: xD↑, xB↓ as RR↑)')
axes[0].legend(); axes[0].grid(alpha=0.3)

nstg_v = np.arange(8,36)
X_t2 = np.tile(base,(len(nstg_v),1)); X_t2[:,3]=nstg_v
xD_t2 = rf_m.predict(X_t2)[:,0]; QC_t2 = rf_m.predict(X_t2)[:,2]
ax2b = axes[1].twinx()
axes[1].plot(nstg_v,xD_t2,'#2A9D8F',lw=2.2,label='xD')
ax2b.plot(nstg_v,QC_t2,'#E76F51',lw=2.2,ls=':',label='QC (kW)')
axes[1].set_xlabel('Number of Stages')
axes[1].set_ylabel('xD',color='#2A9D8F')
ax2b.set_ylabel('QC (kW)',color='#E76F51')
axes[1].set_title('xD & QC vs Stages\n(More stages → better separation)')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/plots/fig6_trends.png',dpi=150,bbox_inches='tight'); plt.close()

# Fig 7 — Residuals
fig,axes = plt.subplots(2,2,figsize=(11,8))
fig.suptitle(f'Residual Analysis — {best}',fontsize=13,fontweight='bold')
for i,(ax,t,lb,c) in enumerate(zip(axes.flat,TARGETS,TLABELS,COLORS)):
    res = y_te[:,i] - yp_b[:,i]
    ax.scatter(yp_b[:,i],res,alpha=0.4,s=15,color=c)
    ax.axhline(0, color='k', linestyle='--', lw=1.2)
    ax.set_xlabel('Predicted',fontsize=9); ax.set_ylabel('Residual',fontsize=9)
    ax.set_title(lb,fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/plots/fig7_residuals.png',dpi=150,bbox_inches='tight'); plt.close()

with open('/home/claude/metrics.json','w') as f: json.dump(results,f,indent=2)

sample = pd.DataFrame()
for i,t in enumerate(TARGETS):
    sample[f'Actual_{t}']    = y_te[:15,i]
    sample[f'Predicted_{t}'] = yp_b[:15,i]
sample.to_csv('/home/claude/sample_predictions.csv',index=False)

print("\nAll plots done.")
print(f"Best model: {best}")
for t in TARGETS:
    m=results[best][t]
    print(f"  {t:35s} R²={m['R2']:.4f} MAE={m['MAE']:.5f} RMSE={m['RMSE']:.5f}")

# Save best model name
with open('/home/claude/best_model.txt','w') as f:
    f.write(best)
    f.write('\n')
    for t in TARGETS:
        m=results[best][t]
        f.write(f"{t}: R2={m['R2']:.4f} MAE={m['MAE']:.6f} RMSE={m['RMSE']:.6f}\n")
