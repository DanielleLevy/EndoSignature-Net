"""Interactive, read-only explorer for the completed EndoSignature-Net study."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explorer import gene_view, load_explorer_data


ROOT = Path(__file__).resolve().parent
COLORS = {"GSE51981": "#5668A9", "GSE212787": "#2A9D8F", "GSE153740": "#D05A6E"}

st.set_page_config(page_title="EndoSignature Explorer", page_icon="🧬", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #fffaf8 0%, #f7f4fb 100%);}
    .hero {padding: 2rem 2.2rem; border-radius: 22px; color: white;
      background: linear-gradient(120deg, #432c61 0%, #8b4269 58%, #d47779 100%);
      box-shadow: 0 14px 35px rgba(67,44,97,.18); margin-bottom: 1.4rem;}
    .hero h1 {font-size: 2.7rem; margin: 0 0 .35rem 0;}
    .hero p {font-size: 1.08rem; max-width: 850px; opacity: .94; margin: 0;}
    .note {background: #fff; border-left: 5px solid #d47779; padding: .85rem 1rem;
      border-radius: 8px; box-shadow: 0 5px 16px rgba(50,30,70,.06);}
    div[data-testid="stMetric"] {background: rgba(255,255,255,.82); border: 1px solid #eadfe8;
      padding: .8rem; border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def data():
    return load_explorer_data(ROOT / "explorer_data")


d = data()
summary = d["project_summary"]
st.markdown(
    f"""<div class="hero"><h1>EndoSignature Explorer</h1>
    <p>Explore a frozen 12-gene endometriosis research signature across single-cell,
    bulk RNA-seq and microarray cohorts—and see why internal stability did not guarantee
    external transfer.</p></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frozen genes", summary["signature_size"])
c2.metric("Studies audited", summary["studies_audited"])
c3.metric("Sample records", summary["sample_records"])
c4.metric("Internal mean ROC-AUC", f"{summary['internal_mean_auc']:.3f}")

tabs = st.tabs(["Study story", "Gene explorer", "Cycle & cells", "Data reality", "Methods & limits"])

with tabs[0]:
    st.subheader("One model, sharply different external outcomes")
    performance = d["cohort_performance"].copy()
    chart = px.bar(
        performance, x="cohort", y="roc_auc", color="cohort", text="roc_auc",
        color_discrete_map=COLORS, custom_data=["analysis", "n_patients", "status"],
    )
    chart.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    chart.update_yaxes(range=[0, 1], title="ROC-AUC")
    chart.update_layout(showlegend=False, height=430, margin=dict(t=20))
    st.plotly_chart(chart, use_container_width=True)
    cols = st.columns(3)
    for col, row in zip(cols, performance.itertuples()):
        col.markdown(f"**{row.cohort}**  \n{row.status}  \n`n={row.n_patients}`")
    st.markdown(
        '<div class="note"><b>Central finding:</b> internally stable coefficients did not guarantee '
        'stable biological directions across cohorts. The negative replication is retained rather '
        'than repaired after observing external outcomes.</div>', unsafe_allow_html=True
    )

with tabs[1]:
    genes = sorted(d["gene_summary"]["gene"].tolist())
    selected = st.selectbox("Choose one of the frozen genes", genes, index=genes.index("SLC1A5"))
    effects, row = gene_view(selected, d["gene_evidence"], d["gene_summary"])
    left, right = st.columns([1.5, 1])
    with left:
        fig = px.bar(
            effects, x="cohort", y="standardized_effect", color="cohort",
            color_discrete_map=COLORS, text="standardized_effect",
            title=f"{selected}: case-control standardized mean difference",
        )
        fig.add_hline(y=0, line_color="#333")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(showlegend=False, height=420, margin=dict(t=55))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.metric("Bootstrap sign stability", f"{row['sign_stability']:.1%}")
        st.metric("Held-out permutation ΔAUC", f"{row['permutation_auc_decrease_mean']:.3f}")
        st.metric("External direction matches", f"{int(row['external_direction_matches'])}/2")
        st.write(f"**Discovery cell family:** {str(row['project_cell_family']).replace('_', ' ')}")
    st.markdown(f"**Literature assessment:** {row['interpretation']}")
    if pd.notna(row.get("primary_reference")):
        st.link_button("Open primary reference", row["primary_reference"])
    st.caption("Predictive importance and association do not establish a causal biomarker.")

with tabs[2]:
    st.subheader("Could cycle phase or broad cell composition explain the gap?")
    left, right = st.columns(2)
    cycle = d["cycle_metrics"].copy()
    cycle["Model"] = cycle["model"].map(
        {"frozen_signature_12": "Frozen signature", "frozen_signature_12_plus_cycle": "Signature + cycle"}
    )
    cycle["Cycle phase"] = cycle["cycle_phase"].str.replace("_", " ")
    with left:
        fig = px.bar(cycle, x="Cycle phase", y="roc_auc", color="Model", barmode="group",
                     color_discrete_sequence=["#5668A9", "#D47779"], title="GSE51981 pooled OOF ROC-AUC")
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=470, margin=dict(t=55))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Adding cycle changed pooled patient-level AUC from 0.877 to 0.875.")
    composition = d["cell_composition"].sort_values("fraction_difference")
    with right:
        fig = px.bar(
            composition, y="cell_family", x="fraction_difference", orientation="h",
            color="fraction_difference", color_continuous_scale=["#5668A9", "#eee8ef", "#D05A6E"],
            color_continuous_midpoint=0, title="Mean fraction difference: endometriosis − control",
            hover_data=["exact_permutation_p", "fdr_bh"],
        )
        fig.update_layout(height=470, coloraxis_showscale=False, margin=dict(t=55))
        fig.update_yaxes(title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Global CLR PERMANOVA: R²=0.094, exact p=0.414; no family passed FDR < 0.05.")
    st.info("Neither recorded cycle phase nor broad cell-family composition emerged as a sufficient single explanation. Both analyses remain sensitivity checks, not causal tests.")

with tabs[3]:
    st.subheader("Many molecular measurements do not guarantee rich clinical metadata")
    completeness = d["metadata_completeness"].pivot(
        index="study_id", columns="field", values="completeness_fraction"
    )
    fig = px.imshow(
        completeness, zmin=0, zmax=1, color_continuous_scale=["#f7f3f5", "#d47779", "#432c61"],
        labels={"color": "Fraction observed"}, aspect="auto",
    )
    fig.update_layout(height=520, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Verified age coverage", "0%")
    b.metric("Verified participant-country coverage", "0%")
    c.metric("Studies with ≥80% cycle coverage", "5 / 8")
    st.warning("Study or laboratory location was not substituted for participant country, and mixed tissue labels were not promoted to clinical disease subtypes.")

with tabs[4]:
    st.subheader("What this explorer is—and is not")
    st.markdown(
        """
        **What was done**
        - Patient-level inference instead of treating cells as independent patients.
        - Sparse single-cell QC and patient-level pseudobulk discovery.
        - Architecture benchmarking followed by a frozen, interpretable L2-logistic model.
        - Locked external application without gene, sign, threshold or hyperparameter retuning.
        - Bootstrap stability, held-out permutation importance and sensitivity audits.

        **Claim boundaries**
        - This is a research signature, not a clinical diagnostic test.
        - Internal performance is not independent validation.
        - Small external cohorts yield wide uncertainty.
        - Cross-platform within-cohort standardization is not deployable single-sample calibration.
        - Gene expression associations do not establish causality.
        """
    )
    st.markdown(f'<div class="note"><b>Project status:</b> {summary["project_status"]}.<br>{summary["claim_boundary"]}</div>', unsafe_allow_html=True)
    st.markdown("[View the source repository](https://github.com/DanielleLevy/EndoSignature-Net)")
