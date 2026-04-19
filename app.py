import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Assessment", layout="wide")

st.title("Corporate Credit Risk Assessment System")
st.write("Multi-year financial analysis (2022–2025) with explainable AI risk scoring")

# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df = df.dropna()

    st.subheader("Raw Data")
    st.dataframe(df)

    # =========================
    # KPI
    # =========================
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg ROA", round(df["ROA"].mean(), 3))
    col2.metric("Avg Debt/Equity", round(df["Debt_to_Equity"].mean(), 2))
    col3.metric("Avg Performance", round(df["Performance_Index"].mean(), 2))

    # =========================
    # COMPANY SUMMARY (2022–2025)
    # =========================
    summary = df.groupby("Company").agg({
        "ROA": "mean",
        "Debt_to_Equity": "mean",
        "Performance_Index": "mean"
    }).reset_index()

    # =========================
    # RISK MODEL
    # =========================
    def risk_model(row):
        score = 0
        reasons = []

        if row["ROA"] < 0.03:
            score += 2
            reasons.append("Low profitability (ROA)")

        if row["Debt_to_Equity"] > 2:
            score += 3
            reasons.append("Very high leverage")

        elif row["Debt_to_Equity"] > 1.5:
            score += 2
            reasons.append("Moderate-high leverage")

        if row["Performance_Index"] < 25:
            score += 2
            reasons.append("Weak performance index")

        return score, reasons

    # =========================
    # BUILD RESULT TABLE
    # =========================
    results = []

    for _, row in summary.iterrows():
        score, reasons = risk_model(row)

        if score <= 1:
            risk = "Low Risk"
        elif score <= 3:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        results.append([
            row["Company"],
            risk,
            score,
            ", ".join(reasons) if reasons else "Stable financial profile"
        ])

    result_df = pd.DataFrame(
        results,
        columns=["Company", "Risk Level", "Risk Score", "Reasons"]
    )

    # =========================
    # RESULTS
    # =========================
    st.subheader("Risk Assessment Results")
    st.dataframe(result_df)

    # =========================
    # TOP 5 SAFEST
    # =========================
    st.subheader("Top 5 Safest Companies")

    top_safe = result_df.sort_values("Risk Score").head(5)
    st.dataframe(top_safe)

    st.bar_chart(top_safe.set_index("Company")["Risk Score"])

    # =========================
    # RISK DISTRIBUTION
    # =========================
    st.subheader("Risk Distribution")

    st.bar_chart(result_df["Risk Level"].value_counts())

    fig, ax = plt.subplots()
    result_df["Risk Level"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    # =========================
    # TREND ANALYSIS
    # =========================
    st.subheader("Financial Trends (2022–2025)")

    company = st.selectbox("Select Company", df["Company"].unique())
    temp = df[df["Company"] == company].sort_values("Year")

    fig2, ax2 = plt.subplots()
    ax2.plot(temp["Year"], temp["ROA"], marker="o", label="ROA")
    ax2.plot(temp["Year"], temp["Debt_to_Equity"], marker="o", label="Debt/Equity")
    ax2.plot(temp["Year"], temp["Performance_Index"], marker="o", label="Performance")
    ax2.set_title(company)
    ax2.legend()

    st.pyplot(fig2)

    # =========================
    # AI CHATBOT
    # =========================
    st.subheader("AI Risk Assistant")

    company_q = st.selectbox("Ask about a company", df["Company"].unique())

    if st.button("Analyze Risk"):
        row = summary[summary["Company"] == company_q].iloc[0]

        score, reasons = risk_model(row)

        st.write("### AI Decision")

        if score <= 1:
            st.success(f"{company_q} → LOW RISK 🟢")
        elif score <= 3:
            st.warning(f"{company_q} → MEDIUM RISK 🟡")
        else:
            st.error(f"{company_q} → HIGH RISK 🔴")

        st.write("### Explanation")
        if reasons:
            for r in reasons:
                st.write("-", r)
        else:
            st.write("No major financial issues detected")

else:
    st.info("Upload your dataset to start analysis")