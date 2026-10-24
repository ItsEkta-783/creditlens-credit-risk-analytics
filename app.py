"""CreditLens interactive model and portfolio analysis app."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.clean_data import OUTPUT_FILE, TARGET, clean, load_raw
from src.modeling import ModelResults, evaluate_predictions, train_models

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="CreditLens | Credit Risk Analytics", page_icon="◈", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
h1, h2, h3 { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; letter-spacing: -0.025em; }
.block-container { padding-top: 1.5rem; max-width: 1440px; }
.hero { background: linear-gradient(115deg,#0d1b2a,#142f42 68%,#18555a); border: 1px solid #274354;
 border-radius: 18px; padding: 1.6rem 1.9rem; color: #eff8fa; margin-bottom: 1.25rem; }
.hero p { color: #bfd3da; margin-bottom: .2rem; }
.eyebrow { color: #80ddd0 !important; text-transform: uppercase; font-weight: 700; letter-spacing: .14em; font-size: .75rem; }
.metric-card { background: #f6f8fa; border: 1px solid #e5e9ed; border-radius: 14px; padding: 1rem 1.15rem; }
.muted { color: #667085; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading the validated CreditLens dataset…")
def load_data() -> pd.DataFrame:
    if OUTPUT_FILE.exists():
        return pd.read_csv(OUTPUT_FILE)
    return clean(load_raw())


@st.cache_resource(show_spinner="Training two models on the stratified training split…")
def get_results(data: pd.DataFrame) -> ModelResults:
    return train_models(data)


@st.cache_data(show_spinner=False)
def global_shap(_model, X_reference: pd.DataFrame, X_sample: pd.DataFrame):
    preprocessor = _model.named_steps["preprocessor"]
    classifier = _model.named_steps["classifier"]
    transformed_sample = preprocessor.transform(X_sample)
    if hasattr(transformed_sample, "toarray"):
        transformed_sample = transformed_sample.toarray()
    names = preprocessor.get_feature_names_out()
    # Tree-path SHAP uses the fitted forest's path statistics and avoids an
    # expensive dense interventional background for this large tree ensemble.
    explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(transformed_sample, check_additivity=False)
    if isinstance(values, list):
        values = values[1]
    elif np.asarray(values).ndim == 3:
        values = np.asarray(values)[:, :, 1]
    return np.asarray(values), np.asarray(names, dtype=str), float(np.asarray(explainer.expected_value).reshape(-1)[-1])


@st.cache_data(show_spinner=False)
def local_shap(_model, X_reference: pd.DataFrame, row: pd.DataFrame):
    preprocessor = _model.named_steps["preprocessor"]
    classifier = _model.named_steps["classifier"]
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(transformed, check_additivity=False)
    if isinstance(values, list):
        values = values[1]
    elif np.asarray(values).ndim == 3:
        values = np.asarray(values)[:, :, 1]
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
    return np.asarray(values).reshape(-1), preprocessor.get_feature_names_out(), expected


def fmt_pct(value):
    return f"{value:.1%}"


data = load_data()
results = get_results(data)
metrics = {name: evaluate_predictions(results.y_test, p) for name, p in results.probabilities.items()}
section = st.sidebar.radio("CREDITLENS", [
    "Executive overview", "Explore the portfolio", "Model comparison", "Risk prediction",
    "SHAP explanations", "Model evaluation", "Limitations & responsible use",
], label_visibility="visible")
st.sidebar.markdown("---")
st.sidebar.caption("Historical data exploration · 2005 Taiwan credit-card clients")
st.sidebar.caption("Models use a stratified 80/20 split · seed 42")

st.markdown('<div class="hero"><p class="eyebrow">CreditLens · Portfolio analytics</p><h1>Understand repayment risk.</h1><p>Explore historical behavior, compare baseline classifiers, and inspect how each prediction was formed.</p></div>', unsafe_allow_html=True)

if section == "Executive overview":
    st.header("Executive overview")
    st.caption("Observed outcomes in the supplied historical dataset. The default rate is descriptive, not a future forecast.")
    n, rate, test_n, positive = len(data), data[TARGET].mean(), len(results.y_test), int(results.y_test.sum())
    cols = st.columns(4)
    for col, value, label in zip(cols, [f"{n:,}", fmt_pct(rate), f"{test_n:,}", fmt_pct(positive / test_n)], ["Clients", "Observed default rate", "Held-out records", "Held-out default share"]):
        with col:
            st.metric(label, value)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        counts = data[TARGET].value_counts().sort_index().rename(index={0: "No default", 1: "Default"}).rename_axis("Outcome").reset_index(name="Clients")
        fig = px.bar(counts, x="Outcome", y="Clients", color="Outcome", color_discrete_map={"No default": "#3b82f6", "Default": "#e07a45"}, text_auto=",")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=20,b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("At a glance")
        st.markdown("""
        - **30,000** cleaned observations with 23 candidate predictors.
        - Identifier `id` and outcome are excluded from model predictors.
        - Missing values and categorical codes are handled inside training-only preprocessing pipelines.
        - Class weighting is used in both models; all reported evaluation is based on held-out predictions.
        """)
        st.info("This application studies next-month default in a historical credit-card dataset. It does not predict loan approval or establish creditworthiness.")

elif section == "Explore the portfolio":
    st.header("Explore the portfolio")
    st.caption("Filters apply to the descriptive charts on this page only; they do not change the training split or evaluation.")
    age_min, age_max = int(data.age.min()), int(data.age.max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))
    choices = st.columns(3)
    sex_values = sorted(data.sex.dropna().unique().tolist())
    edu_values = sorted(data.education.dropna().unique().tolist())
    marital_values = sorted(data.marriage.dropna().unique().tolist())
    with choices[0]: sex_sel = st.multiselect("Sex codes", sex_values, default=sex_values)
    with choices[1]: edu_sel = st.multiselect("Education codes", edu_values, default=edu_values)
    with choices[2]: marital_sel = st.multiselect("Marital status codes", marital_values, default=marital_values)
    filtered = data[data.age.between(*age_range) & data.sex.isin(sex_sel) & data.education.isin(edu_sel) & data.marriage.isin(marital_sel)]
    if filtered.empty:
        st.warning("No records match these filters.")
    else:
        cols = st.columns(3)
        cols[0].metric("Filtered clients", f"{len(filtered):,}")
        cols[1].metric("Observed default rate", fmt_pct(filtered[TARGET].mean()))
        cols[2].metric("Default cases", f"{int(filtered[TARGET].sum()):,}")
        plot_cols = st.columns(2)
        with plot_cols[0]:
            by_age = filtered.assign(age_band=pd.cut(filtered.age, bins=10, duplicates="drop")).groupby("age_band", observed=True)[TARGET].mean().reset_index()
            by_age["age_band"] = by_age.age_band.astype(str)
            st.plotly_chart(px.line(by_age, x="age_band", y=TARGET, markers=True, title="Observed default rate by age band", labels={TARGET:"Default rate", "age_band":"Age band"}), width="stretch")
        with plot_cols[1]:
            by_limit = filtered.assign(limit_band=pd.qcut(filtered.limit_bal, q=8, duplicates="drop")).groupby("limit_band", observed=True)[TARGET].mean().reset_index()
            by_limit["limit_band"] = by_limit.limit_band.astype(str)
            st.plotly_chart(px.bar(by_limit, x="limit_band", y=TARGET, title="Observed default rate by credit-limit band", labels={TARGET:"Default rate", "limit_band":"Credit-limit band"}), width="stretch")
        st.plotly_chart(px.histogram(filtered, x="age", color=TARGET, barmode="overlay", opacity=.72, nbins=25, title="Age distribution by observed outcome", color_discrete_map={0:"#3b82f6",1:"#e07a45"}), width="stretch")

elif section == "Model comparison":
    st.header("Model comparison")
    st.caption("Same stratified split and 0.50 probability threshold. Class weighting helps account for the minority default class; it does not guarantee an optimal operating point.")
    rows = []
    for name, item in metrics.items():
        rows.append({"Model":name, "ROC AUC":item["roc_auc"], "Average precision":item["average_precision"], "Recall":item["recall"], "Precision":item["precision"], "F1":item["f1"], "Brier score ↓":item["brier"]})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"), width="stretch")
    st.markdown("#### Held-out probability distributions")
    model_choice = st.selectbox("Inspect probability distribution", list(results.models))
    dist = pd.DataFrame({"Observed outcome":results.y_test.to_numpy(), "Predicted default probability":results.probabilities[model_choice]})
    st.plotly_chart(px.histogram(dist, x="Predicted default probability", color="Observed outcome", barmode="overlay", opacity=.65, nbins=35, color_discrete_map={0:"#3b82f6",1:"#e07a45"}), width="stretch")
    st.info("Metrics are estimated on a single held-out split. Model comparison is educational and not a validation of use in lending decisions.")

elif section == "Risk prediction":
    st.header("Risk prediction")
    st.caption("Select a record from the held-out test set. Its outcome was not used to fit either model.")
    model_choice = st.selectbox("Model", list(results.models), key="prediction_model")
    record_ids = data.loc[results.X_test.index, "id"].astype(int)
    selected_id = st.selectbox("Held-out record ID", record_ids.tolist())
    selected_index = record_ids[record_ids == selected_id].index[0]
    row = results.X_test.loc[[selected_index]]
    probability = float(results.models[model_choice].predict_proba(row)[0, 1])
    observed = int(results.y_test.loc[selected_index])
    left, right = st.columns([1,1])
    with left:
        st.metric("Estimated probability of next-month default", fmt_pct(probability))
        st.write(f"Model label at 0.50 threshold: **{int(probability >= .5)}**")
        st.write(f"Observed held-out outcome: **{observed}** ({'default' if observed else 'no default'})")
        st.caption("The selected row comes from the test partition. This is a retrospective demonstration and is not a lending recommendation.")
    with right:
        st.dataframe(row.T.rename(columns={selected_index:"Value"}), width="stretch")

elif section == "SHAP explanations":
    st.header("SHAP explanations")
    st.caption("Tree-path SHAP attributes Random Forest predictions to transformed input features using the forest's path statistics. Values explain this model output; they are not causal effects.")
    ids = data.loc[results.X_test.index, "id"].astype(int)
    selected_id = st.selectbox("Held-out record ID", ids.tolist(), key="shap_id")
    selected_index = ids[ids == selected_id].index[0]
    model = results.models["Random Forest"]
    with st.spinner("Calculating SHAP values for a held-out sample…"):
        sample = results.X_test.sample(n=min(75, len(results.X_test)), random_state=42)
        values, names, expected = global_shap(model, results.X_train, sample)
        local_values, local_names, local_expected = local_shap(model, results.X_train, results.X_test.loc[[selected_index]])
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:]
    global_df = pd.DataFrame({"Feature":names[top_idx], "Mean |SHAP value|":mean_abs[top_idx]}).sort_values("Mean |SHAP value|")
    a,b = st.columns(2)
    with a: st.plotly_chart(px.bar(global_df, x="Mean |SHAP value|", y="Feature", orientation="h", title="Global feature importance · mean absolute SHAP"), width="stretch")
    local_df = pd.DataFrame({"Feature":local_names, "SHAP value":local_values}).assign(Magnitude=lambda frame:frame["SHAP value"].abs()).nlargest(15,"Magnitude").sort_values("SHAP value")
    with b: st.plotly_chart(px.bar(local_df, x="SHAP value", y="Feature", orientation="h", color="SHAP value", color_continuous_scale="RdBu_r", title=f"Record {selected_id} · strongest local contributions"), width="stretch")
    p = float(model.predict_proba(results.X_test.loc[[selected_index]])[0,1])
    st.caption(f"For the selected record, the Random Forest probability is {p:.1%}. Global SHAP summarizes a reproducible sample of {len(sample):,} held-out records; local contributions are specific to this record.")

elif section == "Model evaluation":
    st.header("Held-out evaluation")
    model_choice = st.selectbox("Model", list(results.models), key="evaluation_model")
    probs = results.probabilities[model_choice]
    item = evaluate_predictions(results.y_test, probs)
    cols=st.columns(5)
    for col,label,key in zip(cols,["ROC AUC","Avg precision","Recall","Precision","Brier score"],["roc_auc","average_precision","recall","precision","brier"]): col.metric(label, f"{item[key]:.3f}")
    a,b=st.columns(2)
    cm=item["confusion_matrix"]
    with a:
        fig=px.imshow(cm,text_auto=True,x=["Predicted no default","Predicted default"],y=["Actual no default","Actual default"],color_continuous_scale="Blues",title="Confusion matrix · threshold 0.50")
        st.plotly_chart(fig,width="stretch")
    with b:
        fpr,tpr,_=roc_curve(results.y_test,probs)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"ROC AUC {item['roc_auc']:.3f}",line=dict(color="#147d78",width=3))); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Chance",line=dict(dash="dash",color="#98a2b3"))); fig.update_layout(title="ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate",height=370)
        st.plotly_chart(fig,width="stretch")
    a,b=st.columns(2)
    with a:
        precision,recall,_=precision_recall_curve(results.y_test,probs)
        fig=go.Figure(go.Scatter(x=recall,y=precision,mode="lines",line=dict(color="#c56d3b",width=3),name=f"Average precision {item['average_precision']:.3f}")); fig.update_layout(title="Precision–recall curve",xaxis_title="Recall",yaxis_title="Precision",height=370)
        st.plotly_chart(fig,width="stretch")
    with b:
        observed,predicted=calibration_curve(results.y_test,probs,n_bins=10,strategy="quantile")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=predicted,y=observed,mode="lines+markers",name="Model",line=dict(color="#7256a5",width=3))); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect calibration",line=dict(dash="dash",color="#98a2b3"))); fig.update_layout(title=f"Calibration · Brier score {item['brier']:.3f}",xaxis_title="Mean predicted probability",yaxis_title="Observed default frequency",height=370)
        st.plotly_chart(fig,width="stretch")
    st.caption(f"Evaluation uses {len(results.y_test):,} held-out records. Threshold-dependent metrics use 0.50; ROC AUC, average precision, Brier score, ROC, PR, and calibration use held-out probabilities.")

else:
    st.header("Limitations & responsible use")
    st.markdown("""
    ### What this analysis represents
    - The data describes Taiwan credit-card clients and next-month repayment outcomes in 2005. Results may not transfer across time, products, populations, or jurisdictions.
    - The target is **historical next-month credit-card default**. It is not loan-approval prediction, an affordability assessment, or a recommendation to approve or decline credit.
    - These models are portfolio-project baselines trained and evaluated on one random stratified holdout. There is no temporal validation, external validation, monitoring, or deployment review.

    ### How to read the outputs
    - Class weighting changes the training objective. The displayed 0.50 decision threshold is illustrative; precision, recall, calibration, and the costs of errors can trade off.
    - SHAP describes how features contribute to a fitted model's output under its chosen background data. It does not establish causation or explain an individual's circumstances.
    - Sensitive or proxy attributes may encode structural inequities. No fairness assessment or legal review is included here.

    ### Responsible-use boundary
    Do not use these outputs to make or automate decisions about real people. Any consequential application would need appropriate legal and compliance review, representative data, temporal and external validation, fairness analysis, explainability review, security controls, and ongoing monitoring.
    """)

