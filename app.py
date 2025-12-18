import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# ============== Helpers for unpickling ==============
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None): self.cols, self.bounds_ = cols, {}
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=self.cols if self.cols else None)
        for c in df.columns:
            q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
            self.bounds_[c] = (q1 - 1.5*iqr, q3 + 1.5*iqr)
        return self
    def transform(self, X):
        df = pd.DataFrame(X, columns=self.cols if self.cols else None)
        for c in df.columns:
            lo, hi = self.bounds_[c]; df[c] = df[c].clip(lo, hi)
        return df.values

# ============== App Config & Styling ==============
st.set_page_config(page_title="Student Performance & Drop-Out Risk", page_icon="🎓", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#ffffff;
  --soft:#f7f7f8;
  --card:#ffffff;
  --border:#e6e7eb;
  --text:#0f172a;
  --muted:#6b7280;
  --brand:#0ea5e9;
  --brand-weak:#e0f2fe;
  --good:#16a34a;
  --warn:#f59e0b;
  --bad:#dc2626;
  --radius:14px;
}
.block-container { padding-top: 3rem; }
h1,h2,h3 { letter-spacing:.2px; }
.header-bar{
  display:flex; align-items:center; justify-content:space-between;
  padding: .75rem 1rem; border:1px solid var(--border); background:var(--card);
  border-radius: var(--radius); margin-bottom: .6rem;
}
.header-title{display:flex; gap:.6rem; align-items:center;}
.header-chip{font-weight:700; font-size:.85rem; padding:.25rem .6rem; border-radius:999px; background:var(--brand-weak); color:#0369a1; border:1px solid #bae6fd;}
.card{
  padding: 1rem 1.2rem; border:1px solid var(--border); border-radius: var(--radius);
  background: var(--card);
}
.card-muted{ background: var(--soft); border-style:dashed; }
.kpi{
  text-align:center; padding:.9rem .6rem; border:1px dashed var(--border);
  border-radius: 12px; background: var(--soft);
}
.kpi .label{ color:var(--muted); font-size:.9rem; }
.kpi .value{ font-size:1.35rem; margin-top:.15rem; }
.badge{
  display:inline-block; padding: .3rem .7rem; border-radius:999px;
  font-size:.86rem; font-weight:700; border:1px solid;
}
.badge-good{ background:#ecfdf5; color:#065f46; border-color:#a7f3d0; }
.badge-warn{ background:#fffbeb; color:#92400e; border-color:#fde68a; }
.badge-bad { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
.helper{ color:var(--muted); font-size:.9rem; }
.section-title{ font-weight:800; text-transform:uppercase; letter-spacing:.03em; margin:.5rem 0 .5rem; }
.grid{ display:grid; gap:.6rem; grid-template-columns: repeat(12, 1fr); }
.col-6{ grid-column: span 6; }
.col-12{ grid-column: span 12; }
.divider{ height:1px; background:var(--border); margin:.8rem 0; }
.stTabs [data-baseweb="tab"] { font-weight:700; }
.stSlider label, .stSelectbox label, .stNumberInput label, .stTextInput label { font-weight:600; }
.stButton>button { border-radius:10px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ============== Paths & Artifacts ==============
BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
ARTIFACTS_DIR = BASE / "artifacts"

schema = json.load(open(ARTIFACTS_DIR / "schema.json", "r"))
defaults_blob = json.load(open(ARTIFACTS_DIR / "defaults.json", "r"))
defaults = defaults_blob["defaults"]
quick_fields = defaults_blob["quick_fields"]

labels_path = ARTIFACTS_DIR / "labels.json"
try:
    LABELS = json.load(open(labels_path, "r"))
except Exception:
    LABELS = {}

cat_cols = list(schema["categorical"].keys())
num_cols = list(schema["numeric"])
all_cols = cat_cols + num_cols

thr_map = schema.get("thresholds", {"balanced": 0.50, "high_recall": 0.26})
thr_balanced = float(thr_map.get("balanced", 0.50))
thr_highrec  = float(thr_map.get("high_recall", 0.26))

# Models
dropout_pipe = joblib.load(MODEL_DIR / "dropout_model.pkl")
g3_pipe      = joblib.load(MODEL_DIR / "g3_model.pkl")

# G3 defaults & typing
g3_defaults = json.load(open(ARTIFACTS_DIR / "g3_defaults.json", "r"))
g3_all_cols = list(g3_defaults.keys())
def _is_num(v):
    try: float(v); return True
    except Exception: return False
g3_num_cols = [c for c in g3_all_cols if _is_num(g3_defaults[c])]
g3_cat_cols = [c for c in g3_all_cols if c not in g3_num_cols]

# Friendly labels (classification)
MARITAL_MAP = {1:"Single",2:"Married",3:"Widowed",4:"Divorced",5:"Civil union",6:"Legally separated"}
BINARY_LABELS = {
    "Gender": {1:"Male",0:"Female"},
    "Daytimeevening_attendance": {1:"Daytime",0:"Evening"},
    "Displaced": {1:"Yes",0:"No"},
    "Educational_special_needs": {1:"Yes",0:"No"},
    "Debtor": {1:"Yes",0:"No"},
    "Tuition_fees_up_to_date": {1:"Yes",0:"No"},
    "Scholarship_holder": {1:"Yes",0:"No"},
    "International": {1:"Yes",0:"No"},
}
_ORDER_WORDS = {0:"First choice (highest)",1:"Second choice",2:"Third choice",3:"Fourth choice",4:"Fifth choice",
                5:"Sixth choice",6:"Seventh choice",7:"Eighth choice",8:"Ninth choice",9:"Last choice"}
CONTINUOUS_NUMS = {
    "Previous_qualification_grade","Admission_grade",
    "Curricular_units_1st_sem_grade","Curricular_units_2nd_sem_grade",
    "Unemployment_rate","Inflation_rate","GDP"
}
RANGES = {
    "Application_order": (0, 9, 1),
    "Curricular_units_1st_sem_credited": (0, 30, 1),
    "Curricular_units_1st_sem_enrolled": (0, 30, 1),
    "Curricular_units_1st_sem_evaluations": (0, 45, 1),
    "Curricular_units_1st_sem_approved": (0, 30, 1),
    "Curricular_units_1st_sem_without_evaluations": (0, 12, 1),
    "Curricular_units_2nd_sem_credited": (0, 30, 1),
    "Curricular_units_2nd_sem_enrolled": (0, 30, 1),
    "Curricular_units_2nd_sem_evaluations": (0, 40, 1),
    "Curricular_units_2nd_sem_approved": (0, 30, 1),
    "Curricular_units_2nd_sem_without_evaluations": (0, 12, 1),
    "Age_at_enrollment": (15, 70, 1),
    "Previous_qualification_grade": (0.0, 200.0, 0.1),
    "Admission_grade": (0.0, 200.0, 0.1),
    "Curricular_units_1st_sem_grade": (0.0, 20.0, 0.1),
    "Curricular_units_2nd_sem_grade": (0.0, 20.0, 0.1),
    "Unemployment_rate": (0.0, 40.0, 0.1),
    "Inflation_rate": (-5.0, 20.0, 0.1),
    "GDP": (-10.0, 10.0, 0.1),
}
CLS_LABELS = {
    "Gender": ("Gender", "Male or Female"),
    "Age_at_enrollment": ("Age at Enrollment (years)", "Student age when enrolled"),
    "Marital_status": ("Marital Status", "Institution-coded; friendly names shown"),
    "International": ("International Student", "Is the student international?"),
    "Nacionality": ("Nationality (code)", "Institution-coded nationality"),
    "Application_mode": ("Application Mode", "Entry route (phase/contingent)"),
    "Application_order": ("Application Choice Rank (0–9)", "0=first choice (highest), 9=last"),
    "Course": ("Program / Course", "Institution-coded program"),
    "Previous_qualification": ("Previous Qualification (code)", "Highest prior schooling (coded)"),
    "Previous_qualification_grade": ("Previous Qualification Grade (0–200)", "Admission points or prior grade"),
    "Admission_grade": ("Admission Grade (0–200)", "Entry grade/score"),
    "Daytimeevening_attendance": ("Study Period", "Daytime vs Evening"),
    "Debtor": ("Has Outstanding Tuition Debt?", "Yes/No"),
    "Tuition_fees_up_to_date": ("Tuition Paid Up To Date?", "Yes/No"),
    "Scholarship_holder": ("Scholarship Holder?", "Yes/No"),
    "Displaced": ("Displaced Student?", "Yes/No"),
    "Educational_special_needs": ("Educational Special Needs?", "Yes/No"),
    "Mothers_qualification": ("Mother's Education (code)", "Institution-coded"),
    "Fathers_qualification": ("Father's Education (code)", "Institution-coded"),
    "Mothers_occupation": ("Mother's Occupation (code)", "Institution-coded"),
    "Fathers_occupation": ("Father's Occupation (code)", "Institution-coded"),
    "Unemployment_rate": ("Unemployment Rate (%)", "Macro context for the year"),
    "Inflation_rate": ("Inflation Rate (%)", "Macro context for the year"),
    "GDP": ("GDP Growth (%)", "Macro context for the year")
}
def cls_label(col: str) -> str: return CLS_LABELS.get(col, (col.replace("_"," ").title(), ""))[0]
def cls_help(col: str) -> str:  return CLS_LABELS.get(col, ("",""))[1]
def label_for(col: str, code_int: int) -> str:
    custom = LABELS.get(col, {}).get(str(code_int))
    if custom: return custom
    if col == "Marital_status": return MARITAL_MAP.get(code_int, f"Code {code_int}")
    if col == "Application_order": return _ORDER_WORDS.get(code_int, f"Rank {code_int}")
    if col in BINARY_LABELS and code_int in BINARY_LABELS[col]: return BINARY_LABELS[col][code_int]
    if col in ("Mothers_occupation","Fathers_occupation"): return "Unknown / Not working" if code_int == 0 else f"Occupation code {code_int}"
    return f"Code {code_int}"

# ============== G3 helpers ==============
G3_RANGES = {
    "age": (15, 22, 1),
    "traveltime": (1, 4, 1), "studytime": (1, 4, 1),
    "failures": (0, 3, 1),
    "famrel": (1, 5, 1), "freetime": (1, 5, 1), "goout": (1, 5, 1),
    "Dalc": (1, 5, 1), "Walc": (1, 5, 1),
    "health": (1, 5, 1), "absences": (0, 93, 1),
    "Medu": (0, 4, 1), "Fedu": (0, 4, 1),
    "G1": (0, 20, 1), "G2": (0, 20, 1), "G3": (0, 20, 1)
}
G3_CATEGORY = {
    "school":["GP","MS"], "sex":["F","M"], "address":["U","R"], "famsize":["LE3","GT3"], "Pstatus":["T","A"],
    "Mjob":["teacher","health","services","at_home","other"], "Fjob":["teacher","health","services","at_home","other"],
    "reason":["home","reputation","course","other"], "guardian":["mother","father","other"],
    "schoolsup":["yes","no"], "famsup":["yes","no"], "paid":["yes","no"], "activities":["yes","no"],
    "nursery":["yes","no"], "higher":["yes","no"], "internet":["yes","no"], "romantic":["yes","no"]
}
G3_CODED_SELECTS = {
    "traveltime": {1:"< 15 min", 2:"15–30 min", 3:"30–60 min", 4:"> 60 min"},
    "studytime":  {1:"< 2 hrs/week", 2:"2–5 hrs/week", 3:"5–10 hrs/week", 4:"> 10 hrs/week"}
}
G3_LABELS = {
    "school": ("School (GP/MS)", "Student's school: GP or MS"),
    "sex": ("Sex", "F=Female, M=Male"),
    "age": ("Age (years)", "15–22"),
    "address": ("Home Address Type", "U=Urban, R=Rural"),
    "famsize": ("Family Size", "LE3=≤3, GT3=>3 members"),
    "Pstatus": ("Parents' Cohabitation Status", "T=Together, A=Apart"),
    "Medu": ("Mother's Education (0–4)", "0=None, 1=Primary, 2=5th–9th, 3=Secondary, 4=Higher"),
    "Fedu": ("Father's Education (0–4)", "0=None, 1=Primary, 2=5th–9th, 3=Secondary, 4=Higher"),
    "Mjob": ("Mother's Job", "teacher/health/services/at_home/other"),
    "Fjob": ("Father's Job", "teacher/health/services/at_home/other"),
    "reason": ("Reason for Choosing School", "home/reputation/course/other"),
    "guardian": ("Primary Guardian", "mother/father/other"),
    "traveltime": ("Home-to-School Travel Time", "<15m .. >60m (coded 1–4)"),
    "studytime": ("Weekly Study Time", "<2 .. >10 hrs (coded 1–4)"),
    "failures": ("Past Class Failures (0–3+)", "3 means '3 or more'"),
    "schoolsup": ("Extra Educational Support", "yes/no"),
    "famsup": ("Family Educational Support", "yes/no"),
    "paid": ("Extra Paid Classes (math)", "yes/no"),
    "activities": ("Extracurricular Activities", "yes/no"),
    "nursery": ("Attended Nursery School", "yes/no"),
    "higher": ("Wants Higher Education", "yes/no"),
    "internet": ("Internet Access at Home", "yes/no"),
    "romantic": ("In a Romantic Relationship", "yes/no"),
    "famrel": ("Family Relationship Quality (1–5)", "higher is better"),
    "freetime": ("Free Time After School (1–5)", ""),
    "goout": ("Going Out with Friends (1–5)", ""),
    "Dalc": ("Workday Alcohol Consumption (1–5)", ""),
    "Walc": ("Weekend Alcohol Consumption (1–5)", ""),
    "health": ("Current Health Status (1–5)", ""),
    "absences": ("School Absences (0–93)", "number of days"),
    "G1": ("Term 1 Grade (0–20)", ""), "G2": ("Term 2 Grade (0–20)", ""), "G3": ("Final Grade (0–20)", "")
}
def g3_label(col: str) -> str:
    base = G3_LABELS.get(col, (f"[G3] {col}", ""))[0]
    if col in G3_RANGES:
        lo, hi, _ = G3_RANGES[col]; return f"[G3] {base} ({lo}-{hi})"
    return f"[G3] {base}"
def g3_help(col: str) -> str: return G3_LABELS.get(col, ("",""))[1]

def render_g3_field(col: str):
    dv = g3_defaults[col]
    if col in G3_CATEGORY:
        opts = G3_CATEGORY[col]
        try: idx = opts.index(str(dv))
        except ValueError: idx = 0
        return st.selectbox(g3_label(col), opts, index=idx, help=g3_help(col), key=f"g3_cat_{col}")
    if col in G3_CODED_SELECTS:
        mapping = G3_CODED_SELECTS[col]
        codes = list(mapping.keys()); labels = [mapping[c] for c in codes]
        try: default_code = int(dv); idx = codes.index(default_code)
        except Exception: idx = 0
        choice = st.selectbox(g3_label(col), options=list(range(len(codes))),
                              index=idx, format_func=lambda i: labels[i],
                              help=g3_help(col), key=f"g3_code_{col}")
        return codes[choice]
    if col in G3_RANGES:
        lo, hi, step = G3_RANGES[col]
        try: val = int(float(dv))
        except Exception: val = lo
        return st.number_input(g3_label(col), value=int(val), min_value=int(lo), max_value=int(hi),
                               step=int(step), help=g3_help(col), key=f"g3_num_{col}")
    if _is_num(dv):
        return st.number_input(g3_label(col), value=float(dv), help=g3_help(col), key=f"g3_num_{col}")
    return st.text_input(g3_label(col), value=str(dv), help=g3_help(col), key=f"g3_txt_{col}")

# ============== Reset Helpers (Session State) ==============
def reset_dropout_inputs_to_defaults():
    for c in all_cols:
        key = f"class_cat_{c}" if c in cat_cols else f"class_num_{c}"
        val = int(defaults[c]) if c in cat_cols else float(defaults[c])
        st.session_state[key] = val

def reset_g3_inputs_to_defaults():
    for c in g3_all_cols:
        dv = g3_defaults[c]
        if c in G3_CATEGORY:
            st.session_state[f"g3_cat_{c}"] = str(dv)
        elif c in G3_CODED_SELECTS:
            codes = list(G3_CODED_SELECTS[c].keys())
            try:
                idx = codes.index(int(dv))
            except Exception:
                idx = 0
            st.session_state[f"g3_code_{c}"] = idx
        elif c in G3_RANGES or _is_num(dv):
            st.session_state[f"g3_num_{c}"] = int(float(dv))
        else:
            st.session_state[f"g3_txt_{c}"] = str(dv)

def reset_all_inputs():
    reset_dropout_inputs_to_defaults()
    reset_g3_inputs_to_defaults()

# ============== EARLY RESET HOOKS (must run BEFORE any widgets) ==============
if st.session_state.get("_do_reset_all"):
    reset_all_inputs()
    st.session_state.pop("_do_reset_all")
    st.session_state["_reset_toast"] = "All inputs reset to dataset defaults."

if st.session_state.get("_do_reset_dropout"):
    reset_dropout_inputs_to_defaults()
    st.session_state.pop("_do_reset_dropout")
    st.session_state["_reset_toast"] = "Inputs reset to dataset defaults."

if st.session_state.get("_do_reset_g3"):
    reset_g3_inputs_to_defaults()
    st.session_state.pop("_do_reset_g3")
    st.session_state["_reset_toast"] = "[G3] Inputs reset to dataset defaults."

# ============== UI Components ==============
with st.sidebar:
    st.markdown("<div class='header-bar'><div class='header-title'>🎓 <b>Student Outcomes</b></div>", unsafe_allow_html=True)
    st.write("Predict **drop-out risk** and **final grade (G3)**. Use tabs at the top to switch tasks.")
    st.caption("Tip: Quick mode uses 10–12 key inputs; Accurate uses all features.")
    st.markdown("---")
    show_json = st.toggle("Show inputs JSON", value=False)
    allow_download = st.toggle("Enable inputs download", value=False)
    if st.button("Reset All Inputs"):
        st.session_state["_do_reset_all"] = True
        st.rerun()

# Show any deferred toast (after rerun)
if "_reset_toast" in st.session_state:
    st.toast(st.session_state.pop("_reset_toast"), icon="↩️")

# ============== Render helpers — classification ==============
def render_cat(col: str):
    codes = [int(x) for x in schema["categorical"][col]]
    label_map = {c: label_for(col, c) for c in codes}
    default_int = int(defaults[col]); idx = codes.index(default_int) if default_int in codes else 0
    choice = st.selectbox(cls_label(col), options=codes, index=idx,
                          format_func=lambda c: label_map.get(c, f"Code {c}"),
                          help=cls_help(col), key=f"class_cat_{col}")
    return int(choice)

def render_num(col: str):
    if col in CONTINUOUS_NUMS:
        dv = float(defaults[col])
        if col in RANGES:
            lo, hi, step = RANGES[col]
            return st.number_input(cls_label(col), value=float(dv), min_value=float(lo), max_value=float(hi),
                                   step=float(step), help=cls_help(col), key=f"class_num_{col}")
        return st.number_input(cls_label(col), value=float(dv), help=cls_help(col), key=f"class_num_{col}")
    else:
        dv = int(round(float(defaults[col])))
        if col in RANGES:
            lo, hi, step = RANGES[col]
            return st.number_input(cls_label(col), value=int(dv), min_value=int(lo), max_value=int(hi),
                                   step=int(step), help=cls_help(col), key=f"class_num_{col}")
        return st.number_input(cls_label(col), value=int(dv), step=1, help=cls_help(col), key=f"class_num_{col}")

# ============== Header ==============
st.markdown("""
<div class='header-bar'>
  <div class='header-title'>
    <span>🎓</span>
    <div>
      <div style="font-weight:800; line-height:1.1;">Student Performance & Drop-Out Risk</div>
      <div class="helper">Modern, compact interface for quicker decisions.</div>
    </div>
  </div>
  <span class='header-chip'>By Naveen</span>
</div>
""", unsafe_allow_html=True)

# ============== Tabs (Main Tasks) ==============
tab_dropout, tab_g3, tab_about = st.tabs(["Drop-Out Risk", "Final Grade (G3)", "About"])

# -----------------------------------------------------------------------------------
# TAB: DROPOUT
# -----------------------------------------------------------------------------------
with tab_dropout:
    st.markdown("<div class='section-title'>Prediction Setup</div>", unsafe_allow_html=True)
    colA, colB = st.columns([2, 1])

    with colA:
        mode = st.segmented_control("Input Mode", options=["Quick", "Accurate"], default="Quick", help="Quick asks for key features; Accurate exposes all features.")
    
    # Decision Threshold 
    with colA:
        with st.container(border=True):
            st.caption("Decision Threshold")
            thr_presets = {"Balanced": float(thr_balanced), "High Recall": float(thr_highrec), "High Precision": 0.65}
            thr_mode = st.radio("Preset", ["Balanced", "High Recall", "High Precision", "Custom"], horizontal=False, label_visibility="collapsed")
            thr_custom = st.slider("Custom", 0.00, 1.00, float(thr_balanced), 0.01, disabled=(thr_mode != "Custom"))
            thr_val = thr_custom if thr_mode == "Custom" else thr_presets[thr_mode]
            st.caption(f"Current: **{thr_val:.2f}**  ·  Lower → more flagged (↑ recall) · Higher → fewer flagged (↑ precision)")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Student Inputs</div>", unsafe_allow_html=True)

    if mode == "Quick":
        show_cols = quick_fields[:]
        st.markdown("<div class='card card-muted'>Showing key fields; remaining features auto-fill from dataset defaults.</div>", unsafe_allow_html=True)
    else:
        show_cols = all_cols[:]
        st.markdown(f"<div class='card card-muted'>Showing all {len(show_cols)} features.</div>", unsafe_allow_html=True)

    # -------- FORM (Dropout) --------
    with st.form("dropout_form", border=True):
        values = {}
        # render grouped fields
        sections = [
            ("👤 Demographics", ["Gender","Age_at_enrollment","Marital_status","International","Nacionality"]),
            ("🏫 Admissions & Study Mode", ["Application_mode","Application_order","Course","Previous_qualification","Previous_qualification_grade","Admission_grade","Daytimeevening_attendance"]),
            ("💳 Financial / Support", ["Debtor","Tuition_fees_up_to_date","Scholarship_holder","Displaced","Educational_special_needs"]),
            ("📘 1st Semester", [c for c in show_cols if c.startswith("Curricular_units_1st_sem_")]),
            ("📗 2nd Semester", [c for c in show_cols if c.startswith("Curricular_units_2nd_sem_")]),
            ("👪 Parents (coded)", ["Mothers_qualification","Fathers_qualification","Mothers_occupation","Fathers_occupation"]),
            ("🌍 Macro Context", [c for c in ["Unemployment_rate","Inflation_rate","GDP"] if c in show_cols]),
        ]
        cat_cols_local = set(cat_cols)
        def _render_cat(col: str):
            codes = [int(x) for x in schema["categorical"][col]]
            label_map = {c: label_for(col, c) for c in codes}
            default_int = int(defaults[col]); idx = codes.index(default_int) if default_int in codes else 0
            choice = st.selectbox(cls_label(col), options=codes, index=idx,
                                  format_func=lambda c: label_map.get(c, f"Code {c}"),
                                  help=cls_help(col), key=f"class_cat_{col}")
            return int(choice)
        def _render_num(col: str):
            if col in CONTINUOUS_NUMS:
                dv = float(defaults[col])
                if col in RANGES:
                    lo, hi, step = RANGES[col]
                    return st.number_input(cls_label(col), value=float(dv), min_value=float(lo), max_value=float(hi),
                                           step=float(step), help=cls_help(col), key=f"class_num_{col}")
                return st.number_input(cls_label(col), value=float(dv), help=cls_help(col), key=f"class_num_{col}")
            else:
                dv = int(round(float(defaults[col])))
                if col in RANGES:
                    lo, hi, step = RANGES[col]
                    return st.number_input(cls_label(col), value=int(dv), min_value=int(lo), max_value=int(hi),
                                           step=int(step), help=cls_help(col), key=f"class_num_{col}")
                return st.number_input(cls_label(col), value=int(dv), step=1, help=cls_help(col), key=f"class_num_{col}")

        for title, cols in sections:
            cols = [c for c in cols if c in show_cols]
            if not cols: continue
            with st.expander(title, expanded=title in ("👤 Demographics","🏫 Admissions & Study Mode")):
                for c in cols:
                    values[c] = _render_cat(c) if c in cat_cols_local else _render_num(c)

        # fill missing from defaults
        for c in all_cols:
            if c not in values:
                values[c] = int(defaults[c]) if c in cat_cols else float(defaults[c])

        c1, c2 = st.columns([1, 1])
        reset_clicked = c1.form_submit_button("Reset to Defaults", type="secondary")
        predict_clicked = c2.form_submit_button("🔮 Predict", type="primary", use_container_width=True)

        if reset_clicked:
            st.session_state["_do_reset_dropout"] = True
            st.rerun()

    # -------- RESULTS (Dropout) OUTSIDE FORM --------
    if predict_clicked:
        row = {c: (int(values[c]) if c in cat_cols else float(values[c])) for c in all_cols}
        X_df = pd.DataFrame([row], columns=all_cols)
        prob = float(dropout_pipe.predict_proba(X_df)[0, 1])
        pred = int(prob >= thr_val)

        k1, k2, k3 = st.columns([1, 1, 2])
        with k1:
            st.markdown(f"<div class='kpi'><div class='label'>Risk Probability</div><div class='value'>{prob:.3f}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi'><div class='label'>Threshold</div><div class='value'>{thr_val:.2f}</div></div>", unsafe_allow_html=True)
        with k3:
            badge_class = "badge-bad" if pred == 1 else "badge-good"
            label = "Dropout" if pred == 1 else "Non-Dropout"
            st.markdown(f"<div class='card'><span class='badge {badge_class}'>{label}</span> <span class='helper'>Use threshold presets to trade precision/recall.</span></div>", unsafe_allow_html=True)

        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"What-if · Balanced (0.50): **{'Dropout' if prob >= 0.50 else 'Non-Dropout'}**  •  High Recall (0.26): **{'Dropout' if prob >= 0.26 else 'Non-Dropout'}**")

        if show_json:
            st.markdown("**Current Inputs (JSON)**")
            st.json(row, expanded=False)
        if allow_download:
            csv = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")
            st.download_button("Download Inputs (CSV)", csv, file_name="dropout_inputs.csv")

# -----------------------------------------------------------------------------------
# TAB: G3
# -----------------------------------------------------------------------------------
with tab_g3:
    st.markdown("<div class='section-title'>Prediction Setup</div>", unsafe_allow_html=True)
    g3_mode = st.segmented_control("Input Mode", options=["Quick", "Accurate"], default="Quick")

    g3_quick = ["G1", "G2", "studytime", "failures", "absences", "goout", "Dalc", "Walc", "age", "sex", "internet", "freetime"]
    if g3_mode == "Quick":
        g3_show_cols = [c for c in g3_quick if c in g3_all_cols]
        st.markdown("<div class='card card-muted'>[G3] Showing key fields; others use dataset defaults.</div>", unsafe_allow_html=True)
    else:
        g3_show_cols = g3_all_cols[:]
        st.markdown(f"<div class='card card-muted'>[G3] Showing all {len(g3_show_cols)} features.</div>", unsafe_allow_html=True)

    # -------- FORM (G3) --------
    with st.form("g3_form", border=True):
        values_g3 = {}
        with st.expander("📊 Performance Inputs", expanded=True):
            for c in g3_show_cols:
                values_g3[c] = render_g3_field(c)

        # keep omitted defaults
        for c in g3_all_cols:
            if c not in values_g3:
                values_g3[c] = g3_defaults[c]

        c1, c2 = st.columns([1, 1])
        reset_g3 = c1.form_submit_button("Reset to Defaults", type="secondary")
        predict_g3 = c2.form_submit_button("🔮 Predict", type="primary", use_container_width=True)

        if reset_g3:
            st.session_state["_do_reset_g3"] = True
            st.rerun()

    # -------- RESULTS (G3) OUTSIDE FORM --------
    if predict_g3:
        row = {}
        for c in g3_all_cols:
            v = values_g3.get(c, g3_defaults[c])
            if c in g3_num_cols:
                try:
                    row[c] = float(v)
                except Exception:
                    row[c] = float(g3_defaults[c])
            else:
                row[c] = str(v)

        Xg_df = pd.DataFrame([row], columns=g3_all_cols)
        g3_pred = float(g3_pipe.predict(Xg_df)[0])

        band = "Poor" if g3_pred < 8 else ("Average" if g3_pred < 10 else ("Good" if g3_pred < 14 else "Excellent"))
        bclass = "badge-bad" if band == "Poor" else ("badge-warn" if band in ("Average", "Good") else "badge-good")

        k1, k2, k3 = st.columns([1, 1, 2])
        with k1:
            st.markdown(f"<div class='kpi'><div class='label'>Predicted G3</div><div class='value'>{g3_pred:.2f} / 20</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi'><div class='label'>Performance</div><div class='value'>{band}</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='card'><span class='badge {bclass}'>Action</span> <span class='helper'>Use Accurate mode when G1/G2 are missing or uncertain.</span></div>", unsafe_allow_html=True)

        st.progress(min(max(g3_pred / 20.0, 0.0), 1.0))

        if show_json:
            st.markdown("**Current Inputs (JSON)**")
            st.json(row, expanded=False)
        if allow_download:
            csv = pd.DataFrame([row]).to_csv(index=False).encode("utf-8")
            st.download_button("Download [G3] Inputs (CSV)", csv, file_name="g3_inputs.csv")

# -----------------------------------------------------------------------------------
# TAB: ABOUT
# -----------------------------------------------------------------------------------
with tab_about:
    st.markdown("### About this App")
    st.markdown(
        "<div class='card'>"
        "<b>Created By.</b> Naveen<br><br>"
        "<b>Goal.</b> Identify students who may need support (drop-out risk) and estimate final grade (G3) to guide interventions.<br><br>"
        "<b>Models.</b> Tree-based ensemble classifier for Drop-Out risk; ElasticNet regressor for G3.<br><br>"
        "<b>Performance (notebook test set):</b>"
        "<ul>"
        "<li>Drop-Out @ 0.50 → precision≈0.85, recall≈0.76</li>"
        "<li>Drop-Out @ 0.26 → precision≈0.65, recall≈0.90</li>"
        "<li>G3 (ElasticNet): RMSE≈1.16, MAE≈0.72, R²≈0.86</li>"
        "</ul>"
        "<b>Use thresholds wisely:</b> lower for recall; higher for precision.<br><br>"
        "<b>Ethics.</b> Predictions are probabilistic—use to offer support, not penalize. Monitor for bias and track errors over time."
        "</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("#### Files & Configuration")
    st.code(
        "models/\n"
        "  dropout_model.pkl    # binary classifier pipeline (joblib)\n"
        "  g3_model.pkl         # regression pipeline (joblib)\n\n"
        "artifacts/\n"
        "  schema.json          # feature lists & categorical codes\n"
        "  defaults.json        # defaults + quick_fields\n"
        "  g3_defaults.json     # defaults for student-por features\n"
        "  labels.json          # optional friendly labels (overwrite codes)\n",
        language="text"
    )
