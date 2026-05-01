"""
update_madrid_complete.py
=========================
Updates ALL Madrid 2026 prediction files with correct results
through QF, builds SF predictions, and fixes any wrong correct_prediction values.

Uses verified results from the full draw.

Run from your project root:
    python update_madrid_complete.py
"""

import numpy as np
import pandas as pd
import unicodedata
from pathlib import Path
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
import glob, os

REPORTS_DIR  = Path("./reports")
MODEL_PATHS  = ["./models/rf_model.joblib", "./models/rf_model.pkl"]
PROFILE_PATHS = [
    "./reports/player_profiles_post_barcelona_munich_2026.csv",
    "./reports/player_profiles_post_montecarlo_2026.csv",
    "./reports/player_profiles_latest.csv",
]
CAL_PATH = "./models/prob_calibrator.joblib"
SURFACE="Clay"; LEVEL="M"; BEST_OF=3
TEMP_T=1.30; C_SURF=35.0; K_RECENT=6.0; W_INFO=0.25; ELO_LAM=0.15; MKT_LAM=0.20

ALIASES = {
    "Felix Auger-Aliassime":        "Felix Auger Aliassime",
    "Botic Van De Zandschulp":      "Botic van de Zandschulp",
    "Adolfo Daniel Vallejo":        "Diego Vallejo",
    "Martin Damm Jr":               "Martin Damm",
    "Dusan lajovic":                "Dusan Lajovic",
    "Juan Manuel Cerundolo":        "Juan Manuel Cerundolo",
    "Tomas Martin Etcheverry":      "Tomas Martin Etcheverry",
}
def alias(n):
    if not n or pd.isna(n) or n in ("BYE","TBD"): return n
    n = unicodedata.normalize("NFKD",str(n)).encode("ascii","ignore").decode("ascii")
    return ALIASES.get(n,n)

# -- VERIFIED RESULTS (winner always listed first) -------------
# Format: round -> {(pa, pb): (winner, odds_a, odds_b, date)}

ALL_RESULTS = {

"R128": {
    ("Benjamin Bonzi",           "Titouan Droguet"):          ("Benjamin Bonzi",            120,  -143, "2026-04-22"),
    ("Elmer Moller",             "Federico Cina"):            ("Elmer Moller",              -175,   138, "2026-04-22"),
    ("Tomas Machac",             "Francisco Comesana"):       ("Tomas Machac",              -238,   200, "2026-04-22"),
    ("Roberto Bautista Agut",    "Thiago Tirante"):           ("Thiago Tirante",             175,  -200, "2026-04-22"),
    ("Nikoloz Basilashvili",     "Sebastian Ofner"):          ("Sebastian Ofner",            138,  -175, "2026-04-22"),
    ("Lorenzo Sonego",           "Dusan Lajovic"):            ("Dusan Lajovic",              120,  -149, "2026-04-22"),
    ("Alexandre Muller",         "Jan-Lennard Struff"):       ("Jan-Lennard Struff",        -154,   150, "2026-04-22"),
    ("Dusan Prizmic",            "Matteo Berrettini"):        ("Dusan Prizmic",              150,  -143, "2026-04-22"),
    ("Damir Dzumhur",            "Mattia Bellucci"):          ("Damir Dzumhur",              104,  -125, "2026-04-22"),
    ("Ignacio Buse",             "Adrian Mannarino"):         ("Ignacio Buse",              -667,   450, "2026-04-22"),
    ("Zizou Bergs",              "Marin Cilic"):              ("Marin Cilic",               -120,   100, "2026-04-22"),
    ("Jenson Brooksby",          "Emilio Nava"):              ("Emilio Nava",                163,  -200, "2026-04-22"),
    ("Zhizhen Zhang",            "Vit Kopriva"):              ("Vit Kopriva",               -125,   100, "2026-04-22"),
    ("Rafael Jodar",             "Jesper De Jong"):           ("Rafael Jodar",              -588,   400, "2026-04-22"),
    ("Alejandro Tabilo",         "Valentin Royer"):           ("Alejandro Tabilo",          -303,   250, "2026-04-22"),
    ("Jaime Faria",              "Hubert Hurkacz"):           ("Hubert Hurkacz",             220,  -278, "2026-04-22"),
    ("Nicolai Budkov Kjaer",     "Reilly Opelka"):            ("Nicolai Budkov Kjaer",      -108,  -110, "2026-04-23"),
    ("Martin Landaluce",         "Adam Walton"):              ("Adam Walton",               -500,   375, "2026-04-23"),
    ("Marco Trungelliti",        "Daniel Merida"):            ("Daniel Merida",              129,  -149, "2026-04-23"),
    ("Daniel Altmaier",          "Juan Manuel Cerundolo"):    ("Juan Manuel Cerundolo",      125,  -143, "2026-04-23"),
    ("Camilo Ugo Carabelli",     "Gael Monfils"):             ("Camilo Ugo Carabelli",      -189,   174, "2026-04-23"),
    ("Vilius Gaubas",            "Sebastian Baez"):           ("Vilius Gaubas",              220,  -256, "2026-04-23"),
    ("Patrick Kypson",           "Stefanos Tsitsipas"):       ("Stefanos Tsitsipas",         220,  -278, "2026-04-23"),
    ("Diego Vallejo",            "Grigor Dimitrov"):          ("Diego Vallejo",             -110,  -106, "2026-04-23"),
    ("Alexei Popyrin",           "Martin Damm"):              ("Martin Damm",               -179,   163, "2026-04-23"),
    ("Nuno Borges",              "Mariano Navone"):           ("Mariano Navone",             170,  -189, "2026-04-23"),
    ("Jaume Munar",              "Alexander Shevchenko"):     ("Jaume Munar",               -222,   185, "2026-04-23"),
    ("Fabian Marozsan",          "Ethan Quinn"):              ("Fabian Marozsan",           -137,   114, "2026-04-23"),
    ("Terence Atmane",           "Miomir Kecmanovic"):        ("Terence Atmane",             100,  -118, "2026-04-23"),
    ("Cristian Garin",           "Alexander Blockx"):         ("Alexander Blockx",           129,  -149, "2026-04-23"),
    ("Pablo Carreno Busta",      "Marton Fucsovics"):         ("Pablo Carreno Busta",       -125,   100, "2026-04-23"),
    ("Yannick Hanfmann",         "Marcos Giron"):             ("Yannick Hanfmann",          -278,   220, "2026-04-23"),
},

"R64": {
    ("Jannik Sinner",            "Benjamin Bonzi"):           ("Jannik Sinner",             None,  None, "2026-04-24"),
    ("Gabriel Diallo",           "Elmer Moller"):             ("Elmer Moller",              -147,   121, "2026-04-24"),
    ("Cameron Norrie",           "Tomas Machac"):             ("Cameron Norrie",            -147,   120, "2026-04-24"),
    ("Tommy Paul",               "Thiago Tirante"):           ("Thiago Tirante",            -227,   190, "2026-04-24"),
    ("Andrey Rublev",            "Vit Kopriva"):              ("Vit Kopriva",               -333,   262, "2026-04-24"),
    ("Arthur Rinderknech",       "Dusan Lajovic"):            ("Arthur Rinderknech",         167,  -139, "2026-04-24"),
    ("Joao Fonseca",             "Marin Cilic"):              ("Joao Fonseca",              None,  None, "2026-04-24"),
    ("Alex De Minaur",           "Rafael Jodar"):             ("Rafael Jodar",              -118,   110, "2026-04-24"),
    ("Ben Shelton",              "Dusan Prizmic"):            ("Dusan Prizmic",             -213,   179, "2026-04-24"),
    ("Tomas Martin Etcheverry",  "Sebastian Ofner"):          ("Tomas Martin Etcheverry",  -175,   146, "2026-04-24"),
    ("Arthur Fils",              "Ignacio Buse"):             ("Arthur Fils",              -714,   450, "2026-04-24"),
    ("Valentin Vacherot",        "Emilio Nava"):              ("Emilio Nava",               -333,   275, "2026-04-24"),
    ("Jiri Lehecka",             "Alejandro Tabilo"):         ("Jiri Lehecka",              -149,   129, "2026-04-24"),
    ("Alex Michelsen",           "Jan-Lennard Struff"):       ("Alex Michelsen",             118,  -137, "2026-04-24"),
    ("Tallon Griekspoor",        "Damir Dzumhur"):            ("Tallon Griekspoor",         -182,   150, "2026-04-24"),
    ("Lorenzo Musetti",          "Hubert Hurkacz"):           ("Lorenzo Musetti",           -175,   163, "2026-04-24"),
    ("Alexander Bublik",         "Stefanos Tsitsipas"):       ("Stefanos Tsitsipas",        -200,   170, "2026-04-25"),
    ("Corentin Moutet",          "Daniel Merida"):            ("Daniel Merida",             -200,   175, "2026-04-24"),
    ("Alejandro Davidovich Fokina","Pablo Carreno Busta"):    ("Alejandro Davidovich Fokina",-222,  175, "2026-04-25"),
    ("Casper Ruud",              "Jaume Munar"):              ("Casper Ruud",               -250,   250, "2026-04-25"),
    ("Francisco Cerundolo",      "Yannick Hanfmann"):         ("Francisco Cerundolo",       -278,   220, "2026-04-25"),
    ("Luciano Darderi",          "Juan Manuel Cerundolo"):    ("Luciano Darderi",           -161,   138, "2026-04-25"),
    ("Brandon Nakashima",        "Alexander Blockx"):         ("Alexander Blockx",          -161,   137, "2026-04-25"),
    ("Felix Auger Aliassime",    "Vilius Gaubas"):            ("Felix Auger Aliassime",     -357,   311, "2026-04-25"),
    ("Daniil Medvedev",          "Fabian Marozsan"):          ("Daniil Medvedev",           -161,   131, "2026-04-25"),
    ("Denis Shapovalov",         "Nicolai Budkov Kjaer"):     ("Nicolai Budkov Kjaer",      -135,   110, "2026-04-25"),
    ("Learner Tien",             "Diego Vallejo"):            ("Diego Vallejo",             -147,   129, "2026-04-25"),
    ("Flavio Cobolli",           "Camilo Ugo Carabelli"):     ("Flavio Cobolli",            -227,   190, "2026-04-25"),
    ("Karen Khachanov",          "Adam Walton"):              ("Karen Khachanov",           -476,   350, "2026-04-25"),
    ("Jakub Mensik",             "Martin Damm"):              ("Jakub Mensik",              -333,   254, "2026-04-25"),
    ("Ugo Humbert",              "Terence Atmane"):           ("Terence Atmane",            -143,   129, "2026-04-25"),
    ("Alexander Zverev",         "Mariano Navone"):           ("Alexander Zverev",          -455,   333, "2026-04-25"),
},

"R32": {
    ("Jannik Sinner",            "Elmer Moller"):             ("Jannik Sinner",             None,  None, "2026-04-26"),
    ("Cameron Norrie",           "Thiago Tirante"):           ("Cameron Norrie",            -147,   120, "2026-04-26"),
    ("Vit Kopriva",              "Arthur Rinderknech"):       ("Vit Kopriva",               -147,   120, "2026-04-26"),
    ("Joao Fonseca",             "Rafael Jodar"):             ("Rafael Jodar",               120,  -143, "2026-04-26"),
    ("Dusan Prizmic",            "Tomas Martin Etcheverry"):  ("Tomas Martin Etcheverry",    104,  -123, "2026-04-26"),
    ("Arthur Fils",              "Emilio Nava"):              ("Arthur Fils",               -714,   450, "2026-04-26"),
    ("Jiri Lehecka",             "Alex Michelsen"):           ("Jiri Lehecka",              -278,   225, "2026-04-26"),
    ("Tallon Griekspoor",        "Lorenzo Musetti"):          ("Lorenzo Musetti",            326,  -400, "2026-04-26"),
    ("Stefanos Tsitsipas",       "Daniel Merida"):            ("Stefanos Tsitsipas",        -238,   200, "2026-04-27"),
    ("Alejandro Davidovich Fokina","Casper Ruud"):            ("Casper Ruud",                220,  -278, "2026-04-27"),
    ("Francisco Cerundolo",      "Luciano Darderi"):          ("Francisco Cerundolo",       -208,   175, "2026-04-27"),
    ("Alexander Blockx",         "Felix Auger Aliassime"):    ("Alexander Blockx",           250,  -303, "2026-04-27"),
    ("Daniil Medvedev",          "Nicolai Budkov Kjaer"):     ("Daniil Medvedev",           -333,   265, "2026-04-27"),
    ("Diego Vallejo",            "Flavio Cobolli"):           ("Flavio Cobolli",             206,  -227, "2026-04-27"),
    ("Karen Khachanov",          "Jakub Mensik"):             ("Jakub Mensik",               175,  -189, "2026-04-27"),
    ("Terence Atmane",           "Alexander Zverev"):         ("Alexander Zverev",           484,  -714, "2026-04-27"),
},

"R16": {
    ("Jannik Sinner",            "Cameron Norrie"):           ("Jannik Sinner",            -3333,  1800, "2026-04-27"),
    ("Vit Kopriva",              "Rafael Jodar"):             ("Rafael Jodar",               400,  -526, "2026-04-28"),
    ("Tomas Martin Etcheverry",  "Arthur Fils"):              ("Arthur Fils",                254,  -333, "2026-04-27"),
    ("Jiri Lehecka",             "Lorenzo Musetti"):          ("Jiri Lehecka",              -278,   225, "2026-04-27"),
    ("Stefanos Tsitsipas",       "Casper Ruud"):              ("Casper Ruud",                227,  -278, "2026-04-28"),
    ("Francisco Cerundolo",      "Alexander Blockx"):         ("Alexander Blockx",          -250,   210, "2026-04-28"),
    ("Daniil Medvedev",          "Flavio Cobolli"):           ("Flavio Cobolli",            -110,  -108, "2026-04-28"),
    ("Jakub Mensik",             "Alexander Zverev"):         ("Alexander Zverev",           175,  -192, "2026-04-28"),
},

"QF": {
    ("Jannik Sinner",            "Rafael Jodar"):             ("Jannik Sinner",             -625,   450, "2026-04-29"),
    ("Arthur Fils",              "Jiri Lehecka"):             ("Arthur Fils",               -175,   156, "2026-04-29"),
    ("Casper Ruud",              "Alexander Blockx"):         ("Alexander Blockx",          -278,   252, "2026-04-30"),
    ("Flavio Cobolli",           "Alexander Zverev"):         ("Alexander Zverev",           200,  -222, "2026-04-30"),
},
}

# SF draw
SF_DRAW = [
    ("Jannik Sinner",    -625, "Arthur Fils",       500, None, "2026-05-01"),
    ("Alexander Blockx",  304, "Alexander Zverev", -385, None, "2026-05-01"),
]

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

def apply_cal(p, calibrator):
    if calibrator is None: return p
    try: return float(calibrator.predict_proba([[p]])[:,1][0])
    except:
        try: return float(calibrator.predict([p])[0])
        except: return p

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
    out=df.copy(); out["correct_prediction"]=pd.NA; out["correct_prediction_book"]=pd.NA; return out

# -- REFIT PLATT CALIBRATOR ------------------------------------
def refit_platt():
    files=[f for f in glob.glob(f"{REPORTS_DIR}/*_predictions_cck_complete.csv")
           if "_ALL_" not in f and "all_rounds" not in f and "madrid" not in f]
    dfs=[pd.read_csv(f) for f in files]
    df=pd.concat(dfs,ignore_index=True)
    df=df[df["correct_prediction"].notna()&df["prob_player_a_win"].notna()].copy()
    df["prob_player_a_win"]=pd.to_numeric(df["prob_player_a_win"],errors="coerce")
    df["correct_prediction"]=pd.to_numeric(df["correct_prediction"],errors="coerce")
    df=df.dropna(subset=["prob_player_a_win","correct_prediction"])
    pred_a=df["prob_player_a_win"]>=0.5
    df["y"]=np.where(pred_a,df["correct_prediction"],1-df["correct_prediction"])
    X=df["prob_player_a_win"].values.reshape(-1,1); y=df["y"].values
    platt=LogisticRegression(C=1.0,solver="lbfgs")
    platt.fit(X,y)
    dump(platt,CAL_PATH)
    print(f"  Platt calibrator fitted on {len(df):,} samples, saved to {CAL_PATH}")
    print(f"  Smooth curve sample: 0.55->{apply_cal(.55,platt):.3f}  0.60->{apply_cal(.60,platt):.3f}  0.70->{apply_cal(.70,platt):.3f}")
    return platt

# -- UPDATE A ROUND FILE ---------------------------------------
def update_round(round_code, results_dict):
    print(f"\n--- {round_code} ---")
    for suffix in ["_predictions_cck_complete.csv","_predictions_complete.csv"]:
        fpath=REPORTS_DIR/f"madrid2026_{round_code}{suffix}"
        if not fpath.exists(): print(f"  NOT FOUND: {fpath.name}"); continue
        df=pd.read_csv(fpath); fixed=0
        for (pa_raw,pb_raw),(winner,oa,ob,date) in results_dict.items():
            pa,pb,aw=alias(pa_raw),alias(pb_raw),alias(winner)
            # Find row -- try both orderings
            mask=(
                (df["player_a"].apply(alias)==pa)&(df["player_b"].apply(alias)==pb)
            )|(
                (df["player_a"].apply(alias)==pb)&(df["player_b"].apply(alias)==pa)
            )
            if not mask.any(): continue
            idx=df[mask].index[0]
            pred=alias(str(df.at[idx,"pred_winner"]))
            cp=1 if pred==aw else 0
            paf,pbf=devig(ap(oa),ap(ob))
            if not pd.isna(paf):
                bkp=alias(str(df.at[idx,"player_a"])) if paf>=.5 else alias(str(df.at[idx,"player_b"]))
                cpb=1 if bkp==aw else 0
            else:
                cpb=pd.NA
            old_cp=df.at[idx,"correct_prediction"]
            df.at[idx,"correct_prediction"]=cp
            df.at[idx,"correct_prediction_book"]=cpb
            if pd.isna(df.at[idx,"odds_player_a"]) or float(df.at[idx,"odds_player_a"] or 0)==0:
                df.at[idx,"odds_player_a"]=float(oa) if oa else pd.NA
                df.at[idx,"odds_player_b"]=float(ob) if ob else pd.NA
            fixed+=1
            changed="" if (pd.isna(old_cp) or int(old_cp)==cp) else f" (was {int(old_cp)}->{cp})"
            print(f"  {pa[:20]} vs {pb[:20]} -> {aw[:15]} ({'correct' if cp==1 else 'wrong'}){changed}")
        df.to_csv(fpath,index=False); print(f"  {fpath.name}: {fixed} rows updated")
    # Print accuracy
    cck=REPORTS_DIR/f"madrid2026_{round_code}_predictions_cck_complete.csv"
    if cck.exists():
        df=pd.read_csv(cck); cp=df["correct_prediction"].dropna(); cpb=df["correct_prediction_book"].dropna()
        if len(cp): print(f"  {round_code}: model={cp.mean():.1%} ({int(cp.sum())}/{len(cp)}) book={cpb.mean():.1%}")

# -- BUILD SF --------------------------------------------------
def build_sf(prof,pipe,fcols,calibrator):
    print(f"\n--- Building SF ---")
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
        for c in fcols:
            if c not in fd.columns:
                fd[c]={"surface":SURFACE,"tourney_level":LEVEL,"round":"SF","best_of":BEST_OF}.get(c,0)
        p_raw_v=float(pipe.predict_proba(fd[fcols])[:,1][0]) if pipe else elo_p(sa["pre_elo"],sb["pre_elo"])
        p_std=apply_cal(p_raw_v,calibrator)
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
        predc=pa if pc>=.5 else pb; wc,lc=(sa,sb) if predc==pa else(sb,sa)
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
        print(f"  {pa} ({p_std:.1%}) vs {pb} ({1-p_std:.1%}) -- pred:{pred} -- pending")

    sd=pd.DataFrame(sr); cd=pd.DataFrame(cr)
    sd.to_csv(REPORTS_DIR/"madrid2026_SF_predictions_complete.csv",index=False)
    blank(sd).to_csv(REPORTS_DIR/"madrid2026_SF_predictions.csv",index=False)
    cd.to_csv(REPORTS_DIR/"madrid2026_SF_predictions_cck_complete.csv",index=False)
    blank(cd).to_csv(REPORTS_DIR/"madrid2026_SF_predictions_cck.csv",index=False)
    print(f"  SF files written")

def main():
    print("\n=== Madrid 2026 Complete Update ===\n")

    print("--- Refitting Platt calibrator ---")
    calibrator=refit_platt()

    pipe=None; fcols=None
    for p in MODEL_PATHS:
        if Path(p).exists():
            b=load(p); pipe,fcols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else(b,None)
            print(f"Model: {p}"); break
    if not pipe:
        fcols=["diff_avg_rest_days","diff_matches_28d","diff_rolling_10_winrate","diff_rolling_5_winrate",
               "diff_streak","diff_win_rate","diff_peak_elo","diff_current_elo","diff_wr_Clay",
               "diff_wr_Grass","diff_wr_Hard","elo_diff","selo_diff","rank_diff","diff_h2h_wr_prior",
               "surface","tourney_level","best_of","round"]

    prof=load_profiles()

    # Update all completed rounds
    for round_code in ["R128","R64","R32","R16","QF"]:
        if round_code in ALL_RESULTS:
            update_round(round_code, ALL_RESULTS[round_code])

    # Build SF
    build_sf(prof,pipe,fcols,calibrator)

    print("\n\nDone. All rounds updated through QF. SF files created (both pending).")
    print("\nNext:")
    print("  git add reports/madrid2026_*.csv models/prob_calibrator.joblib")
    print("  git commit -m 'update: Madrid complete through QF, SF predictions built'")
    print("  git push")
    print("  python courtiq_engine.py site --output docs/index.html")
    print("  git add docs/index.html && git commit -m 'update: site Madrid SF' && git push")

if __name__=="__main__": main()
