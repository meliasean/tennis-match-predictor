"""
update_madrid_qf_sf.py
=======================
1. Fixes Jodar/Kopriva correct_prediction in R16
2. Fills in QF results (3 complete, 1 still pending today)
3. Builds SF prediction files
4. Also replaces isotonic calibrator with Platt scaling (smooth curve)

Run from your project root:
    python update_madrid_qf_sf.py
"""

import numpy as np
import pandas as pd
import unicodedata
from pathlib import Path
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
import glob, os

REPORTS_DIR = Path("./reports")
MODEL_PATHS = ["./models/rf_model.joblib", "./models/rf_model.pkl"]
PROFILE_PATHS = [
    "./reports/player_profiles_post_barcelona_munich_2026.csv",
    "./reports/player_profiles_post_montecarlo_2026.csv",
    "./reports/player_profiles_latest.csv",
]
CAL_PATH = "./models/prob_calibrator.joblib"
SURFACE  = "Clay"; LEVEL = "M"; BEST_OF = 3
TEMP_T=1.30; C_SURF=35.0; K_RECENT=6.0; W_INFO=0.25; ELO_LAM=0.15; MKT_LAM=0.20

ALIASES = {"Felix Auger-Aliassime":"Felix Auger Aliassime",
           "Botic Van De Zandschulp":"Botic van de Zandschulp"}

def alias(n):
    if not n or pd.isna(n) or n in ("BYE","TBD"): return n
    n = unicodedata.normalize("NFKD",str(n)).encode("ascii","ignore").decode("ascii")
    return ALIASES.get(n,n)

# -- SF DRAW ---------------------------------------------------
SF_DRAW = [
    ("Jannik Sinner",    -625, "Arthur Fils",       500, None,              "2026-05-01"),
    ("Alexander Blockx",  304, "Alexander Zverev", -385, None,              "2026-05-01"),
]

# -- QF RESULTS ------------------------------------------------
QF_RESULTS = {
    ("Jannik Sinner",    "Rafael Jodar"):      ("Jannik Sinner",    -625,  450, "2026-04-29"),
    ("Arthur Fils",      "Jiri Lehecka"):      ("Arthur Fils",      -175,  156, "2026-04-29"),
    ("Casper Ruud",      "Alexander Blockx"):  ("Alexander Blockx", -278,  252, "2026-04-30"),
    ("Flavio Cobolli",   "Alexander Zverev"):  ("Alexander Zverev",  200, -222, "2026-04-30"),
}

# -- PLATT SCALING (smooth calibration, replaces isotonic) ------
def refit_platt_calibrator():
    """
    Fit Platt scaling (logistic regression) instead of isotonic regression.
    Gives a smooth S-curve rather than step function -- no more 14% jumps.
    """
    files = sorted(glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv"))
    files = [f for f in files if "_ALL_" not in f and "all_rounds" not in f]

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["correct_prediction"].notna() & df["prob_player_a_win"].notna()].copy()
    df["prob_player_a_win"] = pd.to_numeric(df["prob_player_a_win"], errors="coerce")
    df["correct_prediction"] = pd.to_numeric(df["correct_prediction"], errors="coerce")
    df = df.dropna(subset=["prob_player_a_win","correct_prediction"])

    # Reconstruct y: did player_a actually win?
    pred_a = df["prob_player_a_win"] >= 0.5
    df["y"] = np.where(pred_a, df["correct_prediction"], 1-df["correct_prediction"])

    X = df["prob_player_a_win"].values.reshape(-1,1)
    y = df["y"].values

    # Platt scaling = logistic regression on raw probabilities
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(X, y)

    # Verify improvement
    from sklearn.metrics import brier_score_loss
    p_raw = X.flatten()
    p_cal = platt.predict_proba(X)[:,1]
    bs_raw = brier_score_loss(y, p_raw)
    bs_cal = brier_score_loss(y, p_cal)
    print(f"  Platt calibrator fitted on {len(df):,} samples")
    print(f"  Brier: raw={bs_raw:.4f} -> platt={bs_cal:.4f} (improvement: {(bs_raw-bs_cal)*100:+.2f}%)")

    # Show smooth curve
    print(f"\n  Smooth calibration curve:")
    print(f"  {'Raw':>6}  {'Calibrated':>10}")
    for p in [0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.65,0.70,0.75,0.80]:
        c = float(platt.predict_proba([[p]])[:,1][0])
        print(f"  {p:.2f}   ->   {c:.3f}   ({(c-p)*100:>+.1f}pp)")

    dump(platt, CAL_PATH)
    print(f"\n  Saved: {CAL_PATH}")
    return platt


def apply_platt(p_raw, calibrator):
    if calibrator is None: return p_raw
    try:
        return float(calibrator.predict_proba([[p_raw]])[:,1][0])
    except:
        # Fallback if it's still the old isotonic
        try: return float(calibrator.predict([p_raw])[0])
        except: return p_raw


# -- UTILITIES -------------------------------------------------
def ap(o):
    if o is None: return np.nan
    try: o=float(o)
    except: return np.nan
    return (-o)/((-o)+100) if o<0 else 100/(o+100)

def devig(a,b):
    if any(np.isnan(v) for v in [a,b]) or a<=0 or b<=0: return np.nan,np.nan
    s=a+b; return a/s,b/s

def logit(p): return np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
def sigmoid(z): return 1/(1+np.exp(-z))
def elo_p(ea,eb): return 1/(1+10**(-(ea-eb)/400))

def cck_calc(p,sa,sb,oa,ob):
    pt=sigmoid(logit(p)/TEMP_T)
    ca,cb=sa.get("cnt_Clay",0),sb.get("cnt_Clay",0)
    ra,rb=sa.get("matches_28d",0),sb.get("matches_28d",0)
    ws=min(1,min(ca,cb)/C_SURF); wr=min(1,min(ra,rb)/K_RECENT)
    ps=0.5+(W_INFO*ws+(1-W_INFO)*wr)*(pt-0.5)
    pe=elo_p(sa["pre_elo"],sb["pre_elo"])
    pm=(1-ELO_LAM)*ps+ELO_LAM*pe
    paf,pbf=devig(ap(oa),ap(ob))
    pc=pm if pd.isna(paf) else (1-MKT_LAM)*pm+MKT_LAM*paf
    return float(np.clip(pc,.01,.99)),float(paf) if not pd.isna(paf) else np.nan,\
           float(pbf) if not pd.isna(pbf) else np.nan,float(pe),float(pt)

def load_profiles():
    for p in PROFILE_PATHS:
        if Path(p).exists():
            df=pd.read_csv(p); df["name"]=df["name"].astype(str).str.strip()
            print(f"  Profiles: {p}"); return df
    raise FileNotFoundError("No profiles")

def snap(prof,player):
    player=alias(player)
    rows=prof[prof["name"]==player]
    D={"pre_elo":1500,"pre_selo":1500,"rest_days":20,"matches_28d":0,
       "rolling_10_winrate":.5,"rolling_5_winrate":.5,"streak":0,
       "avg_rest_days":20,"win_rate":.5,"peak_elo":1500,"current_elo":1500,
       "wr_Clay":.5,"wr_Grass":.5,"wr_Hard":.5,"rank_prior":np.nan,
       "h2h_wr_prior":.5,"cnt_Clay":0,"cnt_Grass":0,"cnt_Hard":0}
    if rows.empty: print(f"    WARN: {player} not in profiles"); return D
    r=rows.iloc[-1]
    def g(c,d=0):
        try: return float(r.get(c,d))
        except: return d
    return {**D,"pre_elo":g("current_elo",1500),"current_elo":g("current_elo",1500),
            "peak_elo":g("peak_elo",1500),"pre_selo":g("selo_Clay",1500),
            "win_rate":g("overall_wr",.5),"avg_rest_days":g("avg_rest_days",20),
            "matches_28d":g("matches_28d",0),"rolling_10_winrate":g("form10_wr",.5),
            "rolling_5_winrate":g("form5_wr",.5),"streak":g("streak",0),
            "wr_Clay":g("wr_Clay",.5),"wr_Grass":g("wr_Grass",.5),"wr_Hard":g("wr_Hard",.5),
            "rest_days":g("avg_rest_days",20)}

BF=["pre_elo","pre_selo","rest_days","matches_28d","rolling_10_winrate","rolling_5_winrate",
    "streak","avg_rest_days","win_rate","peak_elo","current_elo","wr_Clay","wr_Grass","wr_Hard",
    "rank_prior","h2h_wr_prior","cnt_Hard","cnt_Grass","cnt_Clay"]

def d(sa,sb,k): return float(sa.get(k,0))-float(sb.get(k,0)) if sa else 0

def blank(df):
    out=df.copy(); out["correct_prediction"]=pd.NA
    out["correct_prediction_book"]=pd.NA; return out


# -- FIX R16: Jodar/Kopriva ------------------------------------
def fix_r16():
    print("\n--- Fixing R16 correct_prediction values ---")
    for suffix in ["_predictions_cck_complete.csv","_predictions_complete.csv"]:
        fpath = REPORTS_DIR / f"madrid2026_R16{suffix}"
        if not fpath.exists(): continue
        df = pd.read_csv(fpath)
        changed = 0
        for idx, row in df.iterrows():
            pa = alias(str(row.get("player_a","")))
            pb = alias(str(row.get("player_b","")))
            pred = alias(str(row.get("pred_winner","")))
            cp = row.get("correct_prediction")
            if pd.isna(cp): continue

            # Verify correct_prediction is consistent with who actually won
            # We know from results: Jodar beat Kopriva
            # So if pred=Kopriva and cp=1, that's wrong -> should be 0
            known_results = {
                ("Vit Kopriva","Rafael Jodar"):              "Rafael Jodar",
                ("Rafael Jodar","Vit Kopriva"):              "Rafael Jodar",
                ("Jannik Sinner","Cameron Norrie"):          "Jannik Sinner",
                ("Tomas Martin Etcheverry","Arthur Fils"):   "Arthur Fils",
                ("Jiri Lehecka","Lorenzo Musetti"):          "Jiri Lehecka",
                ("Stefanos Tsitsipas","Casper Ruud"):        "Casper Ruud",
                ("Francisco Cerundolo","Alexander Blockx"):  "Alexander Blockx",
                ("Daniil Medvedev","Flavio Cobolli"):        "Flavio Cobolli",
                ("Jakub Mensik","Alexander Zverev"):         "Alexander Zverev",
            }
            actual = known_results.get((pa,pb))
            if actual is None: continue

            correct_cp = 1 if pred == actual else 0
            if int(cp) != correct_cp:
                df.at[idx,"correct_prediction"] = correct_cp
                # Fix book too
                oa = row.get("odds_player_a"); ob = row.get("odds_player_b")
                paf,pbf = devig(ap(oa),ap(ob))
                if not pd.isna(paf):
                    bk_pred = pa if paf>=.5 else pb
                    df.at[idx,"correct_prediction_book"] = 1 if bk_pred==actual else 0
                print(f"  Fixed: {pa} vs {pb} -> actual={actual}, pred={pred}, cp: {int(cp)}->{correct_cp}")
                changed += 1
        df.to_csv(fpath,index=False)
        print(f"  {fpath.name}: {changed} corrections made")

    # Print final R16 accuracy
    cck = REPORTS_DIR/"madrid2026_R16_predictions_cck_complete.csv"
    if cck.exists():
        df=pd.read_csv(cck); cp=df["correct_prediction"].dropna(); cpb=df["correct_prediction_book"].dropna()
        print(f"\n  R16 final: model={cp.mean():.1%} ({int(cp.sum())}/{len(cp)}) book={cpb.mean():.1%}")


# -- UPDATE QF RESULTS -----------------------------------------
def update_qf():
    print("\n--- Updating QF results ---")
    for suffix in ["_predictions_cck_complete.csv","_predictions_complete.csv"]:
        fpath = REPORTS_DIR / f"madrid2026_QF{suffix}"
        if not fpath.exists(): print(f"  NOT FOUND: {fpath.name}"); continue
        df = pd.read_csv(fpath)
        updated = 0
        for (pa_raw,pb_raw),(winner,oa,ob,date) in QF_RESULTS.items():
            pa,pb,aw = alias(pa_raw),alias(pb_raw),alias(winner)
            mask = (
                (df["player_a"].apply(alias)==pa)&(df["player_b"].apply(alias)==pb)
            )|(
                (df["player_a"].apply(alias)==pb)&(df["player_b"].apply(alias)==pa)
            )
            if not mask.any(): print(f"  WARN: not found: {pa} vs {pb}"); continue
            idx=df[mask].index[0]
            pred=alias(str(df.at[idx,"pred_winner"]))
            cp=1 if pred==aw else 0
            paf,pbf=devig(ap(oa),ap(ob))
            bkp=alias(str(df.at[idx,"player_a"])) if (not pd.isna(paf) and paf>=.5) else alias(str(df.at[idx,"player_b"]))
            cpb=1 if bkp==aw else 0
            df.at[idx,"correct_prediction"]=cp
            df.at[idx,"correct_prediction_book"]=cpb
            if pd.isna(df.at[idx,"odds_player_a"]) or df.at[idx,"odds_player_a"]==0:
                df.at[idx,"odds_player_a"]=float(oa)
                df.at[idx,"odds_player_b"]=float(ob)
            updated+=1
            print(f"  {pa} vs {pb} -> {aw} ({'correct' if cp==1 else 'wrong'})")
        df.to_csv(fpath,index=False)
        print(f"  Updated {updated} rows in {fpath.name}")

    cck=REPORTS_DIR/"madrid2026_QF_predictions_cck_complete.csv"
    if cck.exists():
        df=pd.read_csv(cck); cp=df["correct_prediction"].dropna(); cpb=df["correct_prediction_book"].dropna()
        print(f"\n  QF final: model={cp.mean():.1%} ({int(cp.sum())}/{len(cp)}) book={cpb.mean():.1%}")


# -- BUILD SF --------------------------------------------------
def build_sf(prof, pipe, feature_cols, calibrator):
    print("\n--- Building SF ---")
    sr=[]; cr=[]
    for i,(pa_raw,oa,pb_raw,ob,winner,date) in enumerate(SF_DRAW,1):
        pa,pb=alias(pa_raw),alias(pb_raw)
        ho=oa is not None and ob is not None
        oaf=float(oa) if ho else np.nan; obf=float(ob) if ho else np.nan
        sa,sb=snap(prof,pa),snap(prof,pb)
        feats={"surface":SURFACE,"tourney_level":LEVEL,"round":"SF","best_of":BEST_OF}
        for f in BF: feats[f"diff_{f}"]=d(sa,sb,f)
        feats["elo_diff"]=d(sa,sb,"pre_elo"); feats["selo_diff"]=d(sa,sb,"pre_selo")
        feats["rank_diff"]=0.0
        fd=pd.DataFrame([feats])
        for c in feature_cols:
            if c not in fd.columns:
                fd[c]={"surface":SURFACE,"tourney_level":LEVEL,"round":"SF","best_of":BEST_OF}.get(c,0)
        p_raw_v=float(pipe.predict_proba(fd[feature_cols])[:,1][0]) if pipe else elo_p(sa["pre_elo"],sb["pre_elo"])
        p_std=apply_platt(p_raw_v,calibrator)

        pred=pa if p_std>=.5 else pb; conf=max(p_std,1-p_std)
        ws,ls=(sa,sb) if pred==pa else(sb,sa)
        aw=alias(winner) if winner else None
        cp=(1 if alias(pred)==aw else 0) if aw else pd.NA
        paf,pbf=devig(ap(oaf),ap(obf)) if ho else(np.nan,np.nan)
        bkp=pa if(not pd.isna(paf) and paf>=.5) else pb
        cb=(1 if alias(bkp)==aw else 0) if(aw and ho) else pd.NA

        row={"match_no":i,"date":date,"round":"SF",
             "player_a":pa,"odds_player_a":oaf if ho else pd.NA,
             "player_b":pb,"odds_player_b":obf if ho else pd.NA,
             "pred_winner":pred,"correct_prediction":cp,"correct_prediction_book":cb,
             "confidence":round(conf,6),"prob_player_a_win":round(p_std,6),
             "prob_player_b_win":round(1-p_std,6),
             "surface":SURFACE,"tourney_level":LEVEL,"best_of":BEST_OF,
             "book_fair_prob_a":round(paf,6) if not pd.isna(paf) else pd.NA,
             "book_fair_prob_b":round(pbf,6) if not pd.isna(pbf) else pd.NA,
             "p_elo_a":pd.NA,"p_temp_a":pd.NA,
             "delta_elo":d(ws,ls,"pre_elo"),"delta_surface_elo":d(ws,ls,"pre_selo"),
             "delta_peak_elo":d(ws,ls,"peak_elo"),"delta_current_elo":d(ws,ls,"current_elo"),
             "delta_wr_Clay":d(ws,ls,"wr_Clay"),"delta_wr_Grass":d(ws,ls,"wr_Grass"),
             "delta_wr_Hard":d(ws,ls,"wr_Hard"),"delta_win_rate":d(ws,ls,"win_rate"),
             "delta_avg_rest_days":d(ws,ls,"avg_rest_days"),"delta_matches_28d":d(ws,ls,"matches_28d"),
             "delta_rolling_10_winrate":d(ws,ls,"rolling_10_winrate"),
             "delta_rolling_5_winrate":d(ws,ls,"rolling_5_winrate"),
             "delta_streak":d(ws,ls,"streak"),"delta_h2h_wr_prior":d(ws,ls,"h2h_wr_prior"),
             "delta_rank":0.0}
        sr.append(row)

        pc,pafo,pbfo,pe,pt=cck_calc(p_std,sa,sb,oaf,obf)
        predc=pa if pc>=.5 else pb
        wc,lc=(sa,sb) if predc==pa else(sb,sa)
        cpc=(1 if alias(predc)==aw else 0) if aw else pd.NA
        cr.append({**row,"pred_winner":predc,"correct_prediction":cpc,
                   "confidence":round(max(pc,1-pc),6),"prob_player_a_win":round(pc,6),
                   "prob_player_b_win":round(1-pc,6),
                   "book_fair_prob_a":round(pafo,6) if not pd.isna(pafo) else pd.NA,
                   "book_fair_prob_b":round(pbfo,6) if not pd.isna(pbfo) else pd.NA,
                   "p_elo_a":round(pe,6),"p_temp_a":round(pt,6),
                   "delta_elo":d(wc,lc,"pre_elo"),"delta_surface_elo":d(wc,lc,"pre_selo"),
                   "delta_peak_elo":d(wc,lc,"peak_elo"),"delta_current_elo":d(wc,lc,"current_elo"),
                   "delta_wr_Clay":d(wc,lc,"wr_Clay"),"delta_wr_Grass":d(wc,lc,"wr_Grass"),
                   "delta_wr_Hard":d(wc,lc,"wr_Hard"),"delta_win_rate":d(wc,lc,"win_rate"),
                   "delta_avg_rest_days":d(wc,lc,"avg_rest_days"),"delta_matches_28d":d(wc,lc,"matches_28d"),
                   "delta_rolling_10_winrate":d(wc,lc,"rolling_10_winrate"),
                   "delta_rolling_5_winrate":d(wc,lc,"rolling_5_winrate"),
                   "delta_streak":d(wc,lc,"streak"),"delta_h2h_wr_prior":d(wc,lc,"h2h_wr_prior")})

        print(f"  {pa} ({p_std:.1%}) vs {pb} ({1-p_std:.1%})  pred:{pred}  {'pending' if not aw else aw}")

    sd=pd.DataFrame(sr); cd=pd.DataFrame(cr)
    sd.to_csv(REPORTS_DIR/"madrid2026_SF_predictions_complete.csv",index=False)
    blank(sd).to_csv(REPORTS_DIR/"madrid2026_SF_predictions.csv",index=False)
    cd.to_csv(REPORTS_DIR/"madrid2026_SF_predictions_cck_complete.csv",index=False)
    blank(cd).to_csv(REPORTS_DIR/"madrid2026_SF_predictions_cck.csv",index=False)
    print(f"  SF files written (both pending)")


def main():
    print("\n=== Madrid 2026 QF Update + SF Build ===\n")

    # Refit calibrator with Platt scaling (smooth curve, no step jumps)
    print("--- Refitting calibrator (Platt scaling) ---")
    calibrator = refit_platt_calibrator()

    # Load model
    pipe=None; feature_cols=None
    for p in MODEL_PATHS:
        if Path(p).exists():
            b=load(p); pipe,feature_cols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else(b,None)
            print(f"Model: {p}"); break
    if not pipe:
        print("WARNING: ELO fallback")
        feature_cols=["diff_avg_rest_days","diff_matches_28d","diff_rolling_10_winrate",
                      "diff_rolling_5_winrate","diff_streak","diff_win_rate","diff_peak_elo",
                      "diff_current_elo","diff_wr_Clay","diff_wr_Grass","diff_wr_Hard",
                      "elo_diff","selo_diff","rank_diff","diff_h2h_wr_prior",
                      "surface","tourney_level","best_of","round"]

    prof = load_profiles()

    fix_r16()
    update_qf()
    build_sf(prof, pipe, feature_cols, calibrator)

    print("\nDone.")
    print("Next:")
    print("  git add reports/madrid2026_*.csv models/prob_calibrator.joblib")
    print("  git commit -m 'update: Madrid QF results + SF predictions, Platt calibration'")
    print("  git push")
    print("  python courtiq_engine.py site --output docs/index.html && git add docs/index.html && git commit -m 'update: site' && git push")


if __name__ == "__main__":
    main()
