import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="EV Battery Diagnostic", layout="wide")

st.title("🔋 EV Battery Diagnostic System")

uploaded_file = st.file_uploader("📂 Upload Battery CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded Successfully")

    if st.button("🚀 Run Diagnostic"):

        # -------- Feature Engineering --------
        df['V_mean'] = df[[f"V{i}" for i in range(10)]].mean(axis=1)
        df['V_dev'] = df['V0'] - df['V_mean']
        df['Temp_rate'] = df['T0'].diff().fillna(0)

        # -------- Model --------
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(df[['V_dev', 'Temp_rate']])
        df['anomaly'] = model.predict(df[['V_dev', 'Temp_rate']])

        # -------- Explanation Engine --------
        def generate_explanation(fault, v_dev, temp_rate):

            if fault == "Voltage Drop":
                return f"""
Voltage deviation of {round(v_dev,3)}V detected.
Indicates possible Internal Short Circuit (ISC).

Risk: Cell damage & safety hazard.
"""

            elif fault == "Temperature Spike":
                return f"""
Rapid temperature rise ({round(temp_rate,2)}°C).
Indicates Thermal Runaway Risk.

Risk: Overheating / Fire hazard.
"""

            elif fault == "Erratic Signal":
                return """
Irregular signal detected.
Possible sensor fault or loose connection.

Risk: Incorrect system decisions.
"""

            return "Normal operation."

        # -------- Diagnosis --------
        def diagnose(row):

            if row['anomaly'] == -1:

                if abs(row['V_dev']) > 0.2:
                    return (
                        "Voltage Drop",
                        "High",
                        "Isolate affected cell; Activate Limp Home Mode; Limit discharge current; Schedule replacement"
                    )

                elif row['Temp_rate'] > 5:
                    return (
                        "Temperature Spike",
                        "Critical",
                        "Thermal Management Override; Max cooling; Immediate shutdown; Safety alert"
                    )

                else:
                    return (
                        "Erratic Signal",
                        "Medium",
                        "Check wiring; Validate sensors; Use redundant averaging"
                    )

            return ("Normal", "Low", "No action required")

        results = df.apply(lambda row: diagnose(row), axis=1)
        df[['Fault', 'Severity', 'Action']] = pd.DataFrame(results.tolist(), index=df.index)

        anomalies = df[df['anomaly'] == -1]

        # -------- HEALTH SCORE --------
        health_score = 100 - (len(anomalies) / len(df)) * 100

        st.subheader("🔋 Battery Health Meter")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={'text': "Battery Health (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if health_score > 80 else "orange" if health_score > 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}
                ],
            }
        ))

        st.plotly_chart(fig_gauge, use_container_width=True)

        # -------- ALERT --------
        if len(anomalies) > 0:
            st.error("🚨 Real-Time Alert: Battery Fault Detected!")
        else:
            st.success("✅ No active faults detected")

        # -------- DASHBOARD --------
        st.subheader("📊 System Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data Points", len(df))
        col2.metric("Detected Anomalies", len(anomalies))
        col3.metric("Health %", round(health_score, 2))

        # -------- RESULTS --------
        st.subheader("🚨 Diagnostic Results")

        if len(anomalies) > 0:

            for i, row in anomalies.head(5).iterrows():

                explanation = generate_explanation(row['Fault'], row['V_dev'], row['Temp_rate'])

                st.markdown(f"""
---
### ⚠ Fault Detected (Index {i})

**Fault Type:** {row['Fault']}  
**Severity:** {row['Severity']}  

**📌 Explanation:**  
{explanation}

**🛠 Preventive Action:**  
{row['Action']}
""")
        else:
            st.success("No anomalies detected")

        # -------- GRAPH --------
        st.subheader("📈 Battery Behavior")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df['V_dev'], label="Voltage Deviation")

        anomaly_points = df[df['anomaly'] == -1]
        ax.scatter(anomaly_points.index, anomaly_points['V_dev'], color='red', s=30)

        ax.legend()
        st.pyplot(fig)

        # -------- REPORT --------
        st.subheader("📥 Download Report")

        report_text = "EV BATTERY DIAGNOSTIC REPORT\n"
        report_text += "---------------------------------\n\n"

        for i, row in anomalies.iterrows():
            report_text += f"""
Fault Index: {i}
Fault Type: {row['Fault']}
Severity: {row['Severity']}

Explanation:
{generate_explanation(row['Fault'], row['V_dev'], row['Temp_rate'])}

Recommended Action:
{row['Action']}

---------------------------------
"""

        st.download_button(
            "📄 Download Full Diagnostic Report",
            report_text,
            "Battery_Report.txt"
        )