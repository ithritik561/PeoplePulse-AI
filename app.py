import streamlit as st
import pandas as pd
import joblib

# Optional local Ollama. Streamlit Cloud can run without it.
try:
    import ollama
except ImportError:
    ollama = None

st.set_page_config(
    page_title="PeoplePulse AI",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,.05);
    }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    section[data-testid="stSidebar"] { background-color: #111827; }
    section[data-testid="stSidebar"] * { color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data and trained model
data = pd.read_csv("data/employees.csv")
model = joblib.load("models/attrition_model.pkl")
encoders = joblib.load("models/encoders.pkl")


def predict_risk(emp):
    row = emp.drop(labels=["EmployeeID", "Attrition"]).to_frame().T
    for col, encoder in encoders.items():
        row[col] = encoder.transform(row[col])
    return model.predict_proba(row)[0][1]


def risk_level(probability):
    if probability >= 0.60:
        return "🔴 High Risk"
    elif probability >= 0.30:
        return "🟠 Medium Risk"
    return "🟢 Low Risk"


def risk_signals(emp):
    signals = []
    if emp["EngagementScore"] < 60:
        signals.append("Low employee engagement")
    if emp["JobSatisfaction"] <= 2:
        signals.append("Low job satisfaction")
    if emp["WorkLifeBalance"] <= 2:
        signals.append("Poor work-life balance")
    if emp["OverTime"] == "Yes":
        signals.append("Frequent overtime")
    if emp["Workload"] == "High":
        signals.append("High workload")
    if emp["TrainingHours"] < 15:
        signals.append("Low recent training")
    if emp["PromotionYears"] >= 4:
        signals.append("Long time since promotion")
    if emp["AbsenceDays"] >= 10:
        signals.append("High absence days")
    return signals


def action_plan(emp):
    actions = []
    if emp["EngagementScore"] < 60:
        actions.append("Conduct a one-to-one engagement discussion to understand employee concerns.")
    if emp["Workload"] == "High":
        actions.append("Review workload and redistribute tasks where possible.")
    if emp["OverTime"] == "Yes":
        actions.append("Review overtime patterns and identify workload pressure.")
    if emp["TrainingHours"] < 15:
        actions.append("Create a targeted learning and upskilling plan.")
    if emp["JobSatisfaction"] <= 2:
        actions.append("Discuss the main sources of job dissatisfaction.")
    if emp["PromotionYears"] >= 4:
        actions.append("Review career progression and internal mobility options.")
    if emp["AbsenceDays"] >= 10:
        actions.append("Review absence patterns and workplace conditions.")
    return actions


def ai_prompt(emp, eid, pct, signals):
    signal_text = ", ".join(signals) if signals else "None"
    return f"""
You are PeoplePulse AI, an HR decision-support assistant.
Analyze this employee using ONLY these facts.
Employee ID: {eid}
Department: {emp['Department']}
Role: {emp['JobRole']}
Engagement: {emp['EngagementScore']}/100
Job satisfaction: {emp['JobSatisfaction']}/4
Work-life balance: {emp['WorkLifeBalance']}/4
Performance: {emp['PerformanceRating']}
Overtime: {emp['OverTime']}
Training hours: {emp['TrainingHours']}
Workload: {emp['Workload']}
Absence days: {emp['AbsenceDays']}
Years at company: {emp['YearsAtCompany']}
ML attrition risk: {pct}%
Risk signals: {signal_text}
Give a SHORT response with risk summary, 2-3 reasons, 2-3 HR actions, and one 30-day step.
Do not invent facts. Do not recommend automatic firing or rejection.
HR makes the final decision.
"""


def local_ai(prompt):
    """Use local Ollama when available. No API key required."""
    if ollama is None:
        return None
    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 120},
            keep_alive="10m",
        )
        return response["message"]["content"]
    except Exception as exc:
        return f"Ollama Error: {exc}"


def generate_cloud_ai_insight(emp, eid, pct, signals):
    """API-key-free, cloud-safe AI-style insight based only on known data."""
    reasons = []
    if emp["EngagementScore"] < 60:
        reasons.append("Low employee engagement")
    if emp["JobSatisfaction"] <= 2:
        reasons.append("Low job satisfaction")
    if emp["WorkLifeBalance"] <= 2:
        reasons.append("Poor work-life balance")
    if emp["OverTime"] == "Yes":
        reasons.append("Frequent overtime")
    if emp["Workload"] == "High":
        reasons.append("High workload")
    if emp["TrainingHours"] < 15:
        reasons.append("Low recent training")
    if emp["PromotionYears"] >= 4:
        reasons.append("Long time since promotion")
    if emp["AbsenceDays"] >= 10:
        reasons.append("High absence days")

    risk_text = "High" if pct >= 60 else "Medium" if pct >= 30 else "Low"
    reason_text = ", ".join(reasons[:3]) if reasons else "No major workforce risk factors detected"

    action1 = (
        "Review and redistribute workload."
        if emp["Workload"] == "High"
        else "Conduct an HR or manager check-in."
    )
    action2 = (
        "Provide a targeted learning and upskilling plan."
        if emp["TrainingHours"] < 15
        else "Continue monitoring employee engagement."
    )

    return f"""### 🧠 AI-Generated HR Analysis

**Risk Summary**  
Employee **{eid}** has a **{risk_text}** predicted attrition risk of **{pct}%**.

**Key Reasons**  
• {reason_text}

**Recommended HR Actions**  
• {action1}  
• {action2}

**📅 30-Day Action Plan**  
Conduct an HR/manager check-in, apply the selected intervention, monitor engagement, and reassess workforce risk after 30 days.

**👤 Human-in-the-Loop**  
PeoplePulse AI provides decision support only. Final workforce decisions remain with human HR professionals.
"""


# Sidebar
with st.sidebar:
    st.markdown("## 🤖 PeoplePulse AI")
    st.caption("Workforce Decision Intelligence")
    st.divider()

    page = st.radio(
        "Navigate to",
        [
            "🏠 Dashboard",
            "👤 Employee Risk",
            "🤖 AI Action Planner",
            "✨ Generative AI",
            "🔮 What-If Simulator",
            "📊 Workforce Analytics",
        ],
        key="main_navigation",
    )

    st.divider()
    employee_id = st.selectbox(
        "👤 Select Employee",
        data["EmployeeID"].tolist(),
        key="global_employee",
    )
    st.divider()
    st.markdown("### 🎯 AI Pipeline")
    st.write("📊 Data → 🤖 ML Prediction → 🔎 Explain → ⚡ Recommend → 👤 Human Decision")
    st.divider()

    if ollama is not None:
        st.caption("🦙 Local AI: Llama 3.2:3b")
        st.caption("Local Ollama detected")
    else:
        st.caption("🧠 Cloud-compatible mode")
        st.caption("ML + AI-style insights")

    st.caption("No paid API required")
    st.caption("Final decisions remain with HR.")


# Selected employee
employee = data[data["EmployeeID"] == employee_id].iloc[0]
prob = predict_risk(employee)
pct = round(prob * 100, 1)
level = risk_level(prob)
signals = risk_signals(employee)
actions = action_plan(employee)


# Dashboard
if page == "🏠 Dashboard":
    st.title("👥 PeoplePulse AI")
    st.subheader("AI Workforce Early-Warning & Decision Intelligence Platform")
    st.write("Detect workforce risks, understand why they are happening, generate HR recommendations, and simulate possible interventions.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    high_risk_count = sum(predict_risk(row) >= 0.60 for _, row in data.iterrows())
    c1.metric("👥 Total Employees", len(data))
    c2.metric("🚨 High-Risk Employees", high_risk_count)
    c3.metric("📊 Historical Attrition", int((data["Attrition"] == "Yes").sum()))
    c4.metric("💡 Avg Engagement", round(data["EngagementScore"].mean(), 1))

    st.divider()
    st.header("🎯 Selected Employee Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Employee", employee_id)
    c2.metric("ML Attrition Risk", f"{pct}%")
    c3.metric("Risk Level", level)
    st.info("Use the sidebar to open each working module.")

    st.subheader("🚨 Current Risk Signals")
    if signals:
        for signal in signals:
            st.warning("⚠️ " + signal)
    else:
        st.success("✅ No major rule-based risk signals detected.")


# Employee Risk
elif page == "👤 Employee Risk":
    st.title("👤 Employee Risk Analysis")
    st.write("ML-based attrition risk with explainable workforce signals.")
    st.subheader(f"Employee Profile — {employee_id}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Department:**", employee["Department"])
        st.write("**Job Role:**", employee["JobRole"])
        st.write("**Years at Company:**", employee["YearsAtCompany"])
    with c2:
        st.write("**Monthly Income:** ₹", employee["MonthlyIncome"])
        st.write("**Job Satisfaction:**", employee["JobSatisfaction"], "/ 4")
        st.write("**Work-Life Balance:**", employee["WorkLifeBalance"], "/ 4")
    with c3:
        st.write("**Performance Rating:**", employee["PerformanceRating"])
        st.write("**Engagement Score:**", employee["EngagementScore"], "/ 100")
        st.write("**Workload:**", employee["Workload"])

    st.divider()
    st.subheader("🚨 Risk Signals")
    if signals:
        for signal in signals:
            st.warning("⚠️ " + signal)
    else:
        st.success("✅ No major workforce risk signals detected.")

    st.divider()
    st.subheader("🤖 AI Attrition Risk Prediction")
    if prob >= 0.60:
        st.error(f"🔴 High Attrition Risk — Model Probability: {pct}%")
    elif prob >= 0.30:
        st.warning(f"🟠 Medium Attrition Risk — Model Probability: {pct}%")
    else:
        st.success(f"🟢 Low Attrition Risk — Model Probability: {pct}%")
    st.progress(min(max(prob, 0), 1))

    st.divider()
    st.subheader("📋 Employee Data")
    st.dataframe(employee.to_frame().T, use_container_width=True)


# AI Action Planner
elif page == "🤖 AI Action Planner":
    st.title("🤖 AI Action Planner")
    st.write("Convert detected workforce risks into a structured HR action plan.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Employee", employee_id)
    c2.metric("ML Risk", f"{pct}%")
    c3.metric("Risk Level", level)

    st.divider()
    st.subheader("🧠 Why is this employee at risk?")
    if signals:
        for signal in signals:
            st.warning("⚠️ " + signal)
    else:
        st.success("No major risk drivers detected.")

    st.subheader("⚡ Recommended Actions")
    if actions:
        for i, action in enumerate(actions, 1):
            st.info(f"**Action {i}:** {action}")
    else:
        st.success("No immediate intervention required.")

    st.subheader("📅 30-Day Intervention Plan")
    a, b, c = st.columns(3)
    with a:
        st.markdown("### Week 1")
        st.write("🔹 HR / Manager check-in")
        st.write("🔹 Understand employee concerns")
        st.write("🔹 Review workload")
    with b:
        st.markdown("### Week 2–3")
        st.write("🔹 Apply selected intervention")
        st.write("🔹 Provide training/support")
        st.write("🔹 Monitor engagement")
    with c:
        st.markdown("### Week 4")
        st.write("🔹 Reassess employee risk")
        st.write("🔹 Compare new scenario")
        st.write("🔹 HR review")

    st.divider()
    st.subheader("👤 Human-in-the-Loop")
    decision = st.selectbox(
        "HR Review Status",
        ["Select status", "Needs HR Review", "Under Manager Review", "Intervention Planned", "Continue Monitoring"],
        key="planner_hr_status",
    )
    if decision != "Select status":
        st.success(f"✅ HR Status: {decision}")
    st.caption("PeoplePulse AI provides decision support. Final workforce decisions remain with human HR professionals.")


# Generative AI
elif page == "✨ Generative AI":
    st.title("✨ Generative AI HR Insight")
    st.write("Generate an HR analysis from the employee's ML risk, workforce signals and employee data.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Employee", employee_id)
    c2.metric("ML Risk", f"{pct}%")
    c3.metric("Risk Level", level)

    st.divider()
    st.subheader("🔎 Detected Risk Signals")
    if signals:
        for signal in signals:
            st.warning("⚠️ " + signal)
    else:
        st.success("✅ No major risk signals detected.")

    st.divider()
    if st.button("✨ Generate AI HR Insight", type="primary", key="generate_ai_insight"):
        # Actual local Llama when Ollama is available.
        if ollama is not None:
            with st.spinner("🦙 Generating AI insight..."):
                result = local_ai(ai_prompt(employee, employee_id, pct, signals))
            if result and not result.startswith("Ollama Error:"):
                st.markdown("### 🧠 AI-Generated HR Analysis")
                st.markdown(result)
            else:
                st.warning("Local Ollama is unavailable. Showing cloud-safe AI insight instead.")
                st.markdown(generate_cloud_ai_insight(employee, employee_id, pct, signals))
        else:
            # API-key-free cloud fallback based on existing ML/data signals.
            st.markdown(generate_cloud_ai_insight(employee, employee_id, pct, signals))

    st.caption("AI-generated decision support. Final workforce decisions remain with human HR professionals.")


# What-If Simulator
elif page == "🔮 What-If Simulator":
    st.title("🔮 What-If Workforce Simulator")
    st.write("Change workforce conditions and run the same ML model again to compare predicted risk.")
    st.info("💡 Try changing engagement, training hours or workload.")

    c1, c2, c3 = st.columns(3)
    with c1:
        eng = st.slider("Engagement Score", 0, 100, int(employee["EngagementScore"]), key="sim_engagement")
    with c2:
        training = st.slider("Training Hours", 0, 60, int(employee["TrainingHours"]), key="sim_training")
    with c3:
        opts = ["Low", "Medium", "High"]
        workload = st.selectbox("Workload", opts, index=opts.index(employee["Workload"]), key="sim_workload")

    sim = employee.copy()
    sim["EngagementScore"] = eng
    sim["TrainingHours"] = training
    sim["Workload"] = workload
    sim_prob = predict_risk(sim)
    sim_pct = round(sim_prob * 100, 1)

    st.divider()
    st.subheader("📊 Current vs What-If Scenario")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current ML Risk", f"{pct}%")
    c2.metric("Simulated ML Risk", f"{sim_pct}%", delta=f"{sim_pct - pct:.1f}%")
    diff = round(sim_pct - pct, 1)
    with c3:
        if diff < 0:
            st.success(f"📉 Risk Reduced by {abs(diff)}%")
        elif diff > 0:
            st.error(f"📈 Risk Increased by {diff}%")
        else:
            st.info("➡️ Risk Unchanged")

    st.subheader("🧠 Scenario Details")
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Engagement:** {employee['EngagementScore']} → {eng}")
    c2.write(f"**Training Hours:** {employee['TrainingHours']} → {training}")
    c3.write(f"**Workload:** {employee['Workload']} → {workload}")

    st.subheader("💡 AI Interpretation")
    if sim_pct < pct:
        st.success("The simulated intervention reduced the model's predicted attrition risk. HR can review this scenario before considering an actual intervention.")
    elif sim_pct > pct:
        st.warning("The simulated scenario increased the model's predicted attrition risk. HR can explore alternative interventions.")
    else:
        st.info("The simulated changes did not change the model's predicted attrition risk.")

    st.caption("Simulation is a model scenario, not a guarantee of a real-world outcome.")


# Workforce Analytics
elif page == "📊 Workforce Analytics":
    st.title("📊 Workforce Analytics Dashboard")
    st.write("Explore workforce patterns across departments, engagement, performance and historical attrition.")

    summary = (
        data.groupby("Department")
        .agg(
            Employees=("EmployeeID", "count"),
            Average_Engagement=("EngagementScore", "mean"),
            Average_Performance=("PerformanceRating", "mean"),
            Attrition_Count=("Attrition", lambda x: (x == "Yes").sum()),
        )
        .reset_index()
    )

    summary["Attrition_Rate"] = (
        summary["Attrition_Count"] / summary["Employees"] * 100
    ).round(1)
    summary["Average_Engagement"] = summary["Average_Engagement"].round(1)
    summary["Average_Performance"] = summary["Average_Performance"].round(2)

    st.subheader("🏢 Employees by Department")
    st.bar_chart(summary.set_index("Department")["Employees"])

    st.subheader("📈 Average Engagement by Department")
    st.bar_chart(summary.set_index("Department")["Average_Engagement"])

    st.subheader("⚠️ Historical Attrition Rate by Department")
    st.bar_chart(summary.set_index("Department")["Attrition_Rate"])

    st.subheader("🎯 Engagement vs Performance")
    st.scatter_chart(
        data[["EngagementScore", "PerformanceRating"]],
        x="EngagementScore",
        y="PerformanceRating",
    )

    st.subheader("📚 Training Hours vs Engagement")
    st.scatter_chart(
        data[["TrainingHours", "EngagementScore"]],
        x="TrainingHours",
        y="EngagementScore",
    )

    st.subheader("📋 Department Intelligence")
    st.dataframe(summary, use_container_width=True)


# Footer
st.divider()
st.caption("PeoplePulse AI • AI Avengers • AI-powered workforce decision intelligence")
st.caption("Prototype uses synthetic employee data for demonstration purposes.")
