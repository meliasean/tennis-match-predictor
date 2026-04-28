"""
generate_madrid2026.py
======================
Generates all Madrid 2026 prediction CSVs with correct bracket order.

Madrid 96-player draw:
  R128 (32) - qualifying Apr 22-23
  R64  (32) - seeds vs R128 winners Apr 24-25
  R32  (16) - R64 winners play each other Apr 25-27
  R16  (8)  - Apr 27-28 (4 complete, 4 pending)
  QF/SF/F   - blank

Run: python generate_madrid2026.py
Writes 28 files to ./reports/
"""
from __future__ import annotations
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path as _Path

MODEL_PATHS   = ["./models/rf_model.joblib","./models/rf_model.pkl"]
PROFILE_PATHS = [
    "./reports/player_profiles_post_barcelona_munich_2026.csv",
    "./reports/player_profiles_post_montecarlo_2026.csv",
    "./reports/player_profiles_latest.csv",
    "./reports/player_profiles_post_miami_2026.csv",
]
OUT_DIR   = Path("./reports")
SURFACE   = "Clay"; LEVEL = "M"; BEST_OF = 3
TEMP_T    = 1.30;   C_SURF = 35.0; K_RECENT = 6.0
W_INFO    = 0.25;   ELO_LAM = 0.15; MKT_LAM = 0.20
RANK_CLIP = 500

ALIASES = {
    "Felix Auger-Aliassime":       "Felix Auger Aliassime",
    "Botic Van De Zandschulp":     "Botic van de Zandschulp",
    "Jan Lennard Struff":          "Jan-Lennard Struff",
    "Adolfo Daniel Vallejo":       "Diego Vallejo",
}

def alias(n):
    if not n or pd.isna(n) or n in ("BYE","TBD"): return n
    n = unicodedata.normalize("NFKD",str(n)).encode("ascii","ignore").decode("ascii")
    return ALIASES.get(n,n)

# (player_a, odds_a, player_b, odds_b, actual_winner|None, date)
DRAW = {

"R128": [
    ("Benjamin Bonzi",            120, "Titouan Droguet",          -143, "Benjamin Bonzi",              "2026-04-22"),
    ("Elmer Moller",             -175, "Federico Cina",             138,  "Elmer Moller",                "2026-04-22"),
    ("Tomas Machac",             -238, "Francisco Comesana",        200,  "Tomas Machac",                "2026-04-22"),
    ("Roberto Bautista Agut",     175, "Thiago Tirante",           -200,  "Thiago Tirante",              "2026-04-22"),
    ("Nikoloz Basilashvili",      138, "Sebastian Ofner",          -175,  "Sebastian Ofner",             "2026-04-22"),
    ("Lorenzo Sonego",            120, "Dusan Lajovic",            -149,  "Dusan Lajovic",               "2026-04-22"),
    ("Alexandre Muller",         -154, "Jan-Lennard Struff",        150,  "Jan-Lennard Struff",          "2026-04-22"),
    ("Dusan Prizmic",             150, "Matteo Berrettini",        -143,  "Dusan Prizmic",               "2026-04-22"),
    ("Damir Dzumhur",             104, "Mattia Bellucci",          -125,  "Damir Dzumhur",               "2026-04-22"),
    ("Ignacio Buse",             -667, "Adrian Mannarino",          450,  "Ignacio Buse",                "2026-04-22"),
    ("Zizou Bergs",              -120, "Marin Cilic",               100,  "Marin Cilic",                 "2026-04-22"),
    ("Jenson Brooksby",           163, "Emilio Nava",              -200,  "Emilio Nava",                 "2026-04-22"),
    ("Zhizhen Zhang",            -125, "Vit Kopriva",               100,  "Vit Kopriva",                 "2026-04-22"),
    ("Rafael Jodar",             -588, "Jesper De Jong",            400,  "Rafael Jodar",                "2026-04-22"),
    ("Alejandro Tabilo",         -303, "Valentin Royer",            250,  "Alejandro Tabilo",            "2026-04-22"),
    ("Jaime Faria",               220, "Hubert Hurkacz",           -278,  "Hubert Hurkacz",              "2026-04-22"),
    ("Nicolai Budkov Kjaer",     -108, "Reilly Opelka",            -110,  "Nicolai Budkov Kjaer",        "2026-04-23"),
    ("Martin Landaluce",         -500, "Adam Walton",               375,  "Adam Walton",                 "2026-04-23"),
    ("Marco Trungelliti",         129, "Daniel Merida",            -149,  "Daniel Merida",               "2026-04-23"),
    ("Daniel Altmaier",           125, "Juan Manuel Cerundolo",    -143,  "Juan Manuel Cerundolo",       "2026-04-23"),
    ("Camilo Ugo Carabelli",     -189, "Gael Monfils",              174,  "Camilo Ugo Carabelli",        "2026-04-23"),
    ("Vilius Gaubas",             220, "Sebastian Baez",           -256,  "Vilius Gaubas",               "2026-04-23"),
    ("Patrick Kypson",            220, "Stefanos Tsitsipas",       -278,  "Stefanos Tsitsipas",          "2026-04-23"),
    ("Diego Vallejo",            -110, "Grigor Dimitrov",          -106,  "Diego Vallejo",               "2026-04-23"),
    ("Alexei Popyrin",           -179, "Martin Damm",               163,  "Martin Damm",                 "2026-04-23"),
    ("Nuno Borges",               170, "Mariano Navone",           -189,  "Mariano Navone",              "2026-04-23"),
    ("Jaume Munar",              -222, "Alexander Shevchenko",      185,  "Jaume Munar",                 "2026-04-23"),
    ("Fabian Marozsan",          -137, "Ethan Quinn",               114,  "Fabian Marozsan",             "2026-04-23"),
    ("Terence Atmane",            100, "Miomir Kecmanovic",        -118,  "Terence Atmane",              "2026-04-23"),
    ("Cristian Garin",            129, "Alexander Blockx",         -149,  "Alexander Blockx",            "2026-04-23"),
    ("Pablo Carreno Busta",      -125, "Marton Fucsovics",          100,  "Pablo Carreno Busta",         "2026-04-23"),
    ("Yannick Hanfmann",         -278, "Marcos Giron",              220,  "Yannick Hanfmann",            "2026-04-23"),
],

"R64": [
    ("Jannik Sinner",            None, "Benjamin Bonzi",            None, "Jannik Sinner",               "2026-04-24"),
    ("Gabriel Diallo",           -147, "Elmer Moller",               121, "Elmer Moller",                "2026-04-24"),
    ("Cameron Norrie",           -147, "Tomas Machac",               120, "Cameron Norrie",              "2026-04-24"),
    ("Tommy Paul",               -227, "Thiago Tirante",             190, "Thiago Tirante",              "2026-04-24"),
    ("Andrey Rublev",            -333, "Vit Kopriva",                262, "Vit Kopriva",                 "2026-04-24"),
    ("Arthur Rinderknech",        167, "Dusan Lajovic",             -139, "Arthur Rinderknech",          "2026-04-24"),
    ("Joao Fonseca",             None, "Marin Cilic",               None, "Joao Fonseca",                "2026-04-24"),
    ("Alex De Minaur",           -118, "Rafael Jodar",               110, "Rafael Jodar",                "2026-04-24"),
    ("Ben Shelton",              -213, "Dusan Prizmic",              179, "Dusan Prizmic",               "2026-04-24"),
    ("Tomas Martin Etcheverry",  -175, "Sebastian Ofner",            146, "Tomas Martin Etcheverry",     "2026-04-24"),
    ("Arthur Fils",              -714, "Ignacio Buse",               450, "Arthur Fils",                 "2026-04-24"),
    ("Valentin Vacherot",        -333, "Emilio Nava",                275, "Emilio Nava",                 "2026-04-24"),
    ("Jiri Lehecka",             -149, "Alejandro Tabilo",           129, "Jiri Lehecka",                "2026-04-24"),
    ("Alex Michelsen",            118, "Jan-Lennard Struff",        -137, "Alex Michelsen",              "2026-04-24"),
    ("Tallon Griekspoor",        -182, "Damir Dzumhur",              150, "Tallon Griekspoor",           "2026-04-24"),
    ("Lorenzo Musetti",          -175, "Hubert Hurkacz",             163, "Lorenzo Musetti",             "2026-04-24"),
    ("Alexander Bublik",         -200, "Stefanos Tsitsipas",         170, "Stefanos Tsitsipas",          "2026-04-25"),
    ("Corentin Moutet",          -200, "Daniel Merida",              175, "Daniel Merida",               "2026-04-24"),
    ("Alejandro Davidovich Fokina",-222,"Pablo Carreno Busta",       175, "Alejandro Davidovich Fokina", "2026-04-25"),
    ("Casper Ruud",              -250, "Jaume Munar",                250, "Casper Ruud",                 "2026-04-25"),
    ("Francisco Cerundolo",      -278, "Yannick Hanfmann",           220, "Francisco Cerundolo",         "2026-04-25"),
    ("Luciano Darderi",          -161, "Juan Manuel Cerundolo",      138, "Luciano Darderi",             "2026-04-25"),
    ("Brandon Nakashima",        -161, "Alexander Blockx",           137, "Alexander Blockx",            "2026-04-25"),
    ("Felix Auger Aliassime",    -357, "Vilius Gaubas",              311, "Felix Auger Aliassime",       "2026-04-25"),
    ("Daniil Medvedev",          -161, "Fabian Marozsan",            131, "Daniil Medvedev",             "2026-04-25"),
    ("Denis Shapovalov",         -135, "Nicolai Budkov Kjaer",       110, "Nicolai Budkov Kjaer",        "2026-04-25"),
    ("Learner Tien",             -147, "Diego Vallejo",              129, "Diego Vallejo",               "2026-04-25"),
    ("Flavio Cobolli",           -227, "Camilo Ugo Carabelli",       190, "Flavio Cobolli",              "2026-04-25"),
    ("Karen Khachanov",          -476, "Adam Walton",                350, "Karen Khachanov",             "2026-04-25"),
    ("Jakub Mensik",             -333, "Martin Damm",                254, "Jakub Mensik",                "2026-04-25"),
    ("Ugo Humbert",              -143, "Terence Atmane",             129, "Terence Atmane",              "2026-04-25"),
    ("Alexander Zverev",         -455, "Mariano Navone",             333, "Alexander Zverev",            "2026-04-25"),
],

"R32": [
    ("Jannik Sinner",            None, "Elmer Moller",              None, "Jannik Sinner",               "2026-04-26"),
    ("Cameron Norrie",           -147, "Thiago Tirante",             120, "Cameron Norrie",              "2026-04-26"),
    ("Vit Kopriva",              -147, "Arthur Rinderknech",         120, "Vit Kopriva",                 "2026-04-26"),
    ("Joao Fonseca",              120, "Rafael Jodar",              -143, "Rafael Jodar",                "2026-04-26"),
    ("Dusan Prizmic",             104, "Tomas Martin Etcheverry",  -123,  "Tomas Martin Etcheverry",     "2026-04-26"),
    ("Arthur Fils",              -714, "Emilio Nava",               450,  "Arthur Fils",                 "2026-04-26"),
    ("Jiri Lehecka",             -278, "Alex Michelsen",            225,  "Jiri Lehecka",                "2026-04-26"),
    ("Tallon Griekspoor",         326, "Lorenzo Musetti",          -400,  "Lorenzo Musetti",             "2026-04-26"),
    ("Stefanos Tsitsipas",       -238, "Daniel Merida",             200,  "Stefanos Tsitsipas",          "2026-04-27"),
    ("Alejandro Davidovich Fokina", 220,"Casper Ruud",             -278,  "Casper Ruud",                 "2026-04-27"),
    ("Francisco Cerundolo",      -208, "Luciano Darderi",           175,  "Francisco Cerundolo",         "2026-04-27"),
    ("Alexander Blockx",          250, "Felix Auger Aliassime",    -303,  "Alexander Blockx",            "2026-04-27"),
    ("Daniil Medvedev",          -333, "Nicolai Budkov Kjaer",      265,  "Daniil Medvedev",             "2026-04-27"),
    ("Diego Vallejo",             206, "Flavio Cobolli",           -227,  "Flavio Cobolli",              "2026-04-27"),
    ("Karen Khachanov",           175, "Jakub Mensik",             -189,  "Jakub Mensik",                "2026-04-27"),
    ("Terence Atmane",            484, "Alexander Zverev",         -714,  "Alexander Zverev",            "2026-04-27"),
],

"R16": [
    ("Jannik Sinner",           -3333, "Cameron Norrie",            1800, "Jannik Sinner",               "2026-04-27"),
    ("Vit Kopriva",               400, "Rafael Jodar",              -526, "Vit Kopriva",                 "2026-04-28"),
    ("Tomas Martin Etcheverry",   254, "Arthur Fils",               -333, "Arthur Fils",                 "2026-04-27"),
    ("Jiri Lehecka",             -278, "Lorenzo Musetti",            225, "Jiri Lehecka",                "2026-04-27"),
    ("Stefanos Tsitsipas",        227, "Casper Ruud",               -278, None,                          "2026-04-28"),
    ("Francisco Cerundolo",      -250, "Alexander Blockx",           200, None,                          "2026-04-28"),
    ("Daniil Medvedev",          -125, "Flavio Cobolli",             106, None,                          "2026-04-28"),
    ("Jakub Mensik",              175, "Alexander Zverev",          -213, None,                          "2026-04-28"),
],

"QF": [
    ("Jannik Sinner",            None, "TBD",                       None, None, "2026-04-29"),
    ("Arthur Fils",              None, "TBD",                       None, None, "2026-04-29"),
    ("TBD",                      None, "TBD",                       None, None, "2026-04-30"),
    ("TBD",                      None, "TBD",                       None, None, "2026-04-30"),
],

"SF": [
    ("TBD", None, "TBD", None, None, "2026-05-01"),
    ("TBD", None, "TBD", None, None, "2026-05-01"),
],

"F": [
    ("TBD", None, "TBD", None, None, "2026-05-03"),
],
}

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
    return float(np.clip(pc,.01,.99)),float(paf) if not pd.isna(paf) else np.nan,float(pbf) if not pd.isna(pbf) else np.nan,float(pe),float(pt)

def load_prof():
    for p in PROFILE_PATHS:
        if Path(p).exists():
            df=pd.read_csv(p); df["name"]=df["name"].astype(str).str.strip()
            print(f"  Profiles: {p} ({len(df)} players)"); return df
    raise FileNotFoundError(str(PROFILE_PATHS))

def snap(prof,player):
    player=alias(player)
    rows=prof[prof["name"]==player]
    D={"pre_elo":1500,"pre_selo":1500,"rest_days":20,"matches_28d":0,
       "rolling_10_winrate":.5,"rolling_5_winrate":.5,"streak":0,
       "avg_rest_days":20,"win_rate":.5,"peak_elo":1500,"current_elo":1500,
       "wr_Clay":.5,"wr_Grass":.5,"wr_Hard":.5,"rank_prior":np.nan,
       "h2h_wr_prior":.5,"cnt_Clay":0,"cnt_Grass":0,"cnt_Hard":0}
    if rows.empty:
        print(f"    WARN: {player} not in profiles"); return D
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

def build(rnd,data,prof,pipe,fcols,calibrator=None):
    sr=[]; cr=[]
    for i,(pa_r,oa,pb_r,ob,winner,date) in enumerate(data,1):
        pa,pb=alias(pa_r),alias(pb_r)
        tbd=pa in("TBD","BYE") or pb in("TBD","BYE")
        ho=oa is not None and ob is not None and not tbd
        oaf=float(oa) if ho else np.nan; obf=float(ob) if ho else np.nan
        sa,sb=({},{}) if tbd else (snap(prof,pa),snap(prof,pb))
        if tbd:
            p_std=0.5
        else:
            feats={"surface":SURFACE,"tourney_level":LEVEL,"round":rnd,"best_of":BEST_OF}
            for f in BF: feats[f"diff_{f}"]=d(sa,sb,f)
            feats["elo_diff"]=d(sa,sb,"pre_elo"); feats["selo_diff"]=d(sa,sb,"pre_selo")
            ra,rb=sa.get("rank_prior",np.nan),sb.get("rank_prior",np.nan)
            feats["rank_diff"]=0 if(pd.isna(ra) or pd.isna(rb)) else float(np.clip(ra-rb,-RANK_CLIP,RANK_CLIP))
            fd=pd.DataFrame([feats])
            for c in fcols:
                if c not in fd.columns:
                    fd[c]={"surface":SURFACE,"tourney_level":LEVEL,"round":rnd,"best_of":BEST_OF}.get(c,0)
            p_raw=float(pipe.predict_proba(fd[fcols])[:,1][0]) if pipe else elo_p(sa["pre_elo"],sb["pre_elo"])
            p_std=float(calibrator.predict([p_raw])[0]) if calibrator else p_raw
        pred=pa if p_std>=.5 else pb; conf=max(p_std,1-p_std)
        ws,ls=(sa,sb) if pred==pa and not tbd else (sb,sa) if not tbd else ({},{})
        aw=alias(winner) if winner else None
        cp=(1 if alias(pred)==aw else 0) if aw else pd.NA
        paf,pbf=devig(ap(oaf),ap(obf)) if ho else(np.nan,np.nan)
        bkp=pa if(not pd.isna(paf) and paf>=.5) else pb
        cb=(1 if alias(bkp)==aw else 0) if(aw and ho) else pd.NA
        row={"match_no":i,"date":date,"round":rnd,
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
        pc,pafo,pbfo,pe,pt=cck_calc(p_std,sa,sb,oaf,obf) if not tbd else(0.5,np.nan,np.nan,0.5,0.5)
        predc=pa if pc>=.5 else pb; wc,lc=(sa,sb) if predc==pa and not tbd else(sb,sa) if not tbd else({},{})
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
    return pd.DataFrame(sr),pd.DataFrame(cr)

def blank(df):
    out=df.copy(); out["correct_prediction"]=pd.NA; out["correct_prediction_book"]=pd.NA; return out

def main():
    print("\n=== Madrid 2026 Prediction Generator ===\n")
    pipe=None; fcols=None
    for p in MODEL_PATHS:
        if Path(p).exists():
            b=load(p); pipe,fcols=(b["pipeline"],b["feature_cols"]) if isinstance(b,dict) else(b,None)
            print(f"Model loaded: {p}"); break
    # Load calibrator if available
    cal_path = './models/prob_calibrator.joblib'
    calibrator = load(cal_path) if _Path(cal_path).exists() else None
    if calibrator:
        print(f'  Calibrator loaded: {cal_path}')
    else:
        print('  WARNING: No calibrator found -- using raw probabilities')

    if not pipe:
        print("WARNING: Model not found — ELO fallback")
        fcols=["diff_avg_rest_days","diff_matches_28d","diff_rolling_10_winrate","diff_rolling_5_winrate",
               "diff_streak","diff_win_rate","diff_peak_elo","diff_current_elo","diff_wr_Clay","diff_wr_Grass",
               "diff_wr_Hard","elo_diff","selo_diff","rank_diff","diff_h2h_wr_prior","surface","tourney_level","best_of","round"]
    print(); prof=load_prof(); print(); tot_s=tot_c=tot_m=0
    for rnd,data in DRAW.items():
        print(f"Processing {rnd} ({len(data)} matches)...")
        sd,cd=build(rnd,data,prof,pipe,fcols,calibrator)
        stem=f"madrid2026_{rnd}"
        sd.to_csv(OUT_DIR/f"{stem}_predictions_complete.csv",index=False)
        blank(sd).to_csv(OUT_DIR/f"{stem}_predictions.csv",index=False)
        cd.to_csv(OUT_DIR/f"{stem}_predictions_cck_complete.csv",index=False)
        blank(cd).to_csv(OUT_DIR/f"{stem}_predictions_cck.csv",index=False)
        cp=sd["correct_prediction"].dropna(); cpb=sd["correct_prediction_book"].dropna()
        ccp=cd["correct_prediction"].dropna()
        if len(cp):
            print(f"  STD={cp.mean():.1%}  book={cpb.mean():.1%}  CCK={ccp.mean():.1%}  ({len(cp)} scored)")
            tot_s+=int(cp.sum()); tot_c+=int(ccp.sum()); tot_m+=len(cp)
        else:
            print(f"  Blank — results pending")
    print(f"\n{'-'*50}")
    if tot_m: print(f"STD: {tot_s}/{tot_m} ({tot_s/tot_m:.1%})  CCK: {tot_c}/{tot_m} ({tot_c/tot_m:.1%})")
    print(f"28 files -> {OUT_DIR.resolve()}\n")

if __name__=="__main__": main()
