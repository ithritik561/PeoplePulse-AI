import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# OPTIONAL OLLAMA
# --------------------------------------------------

try:
    import ollama
except ImportError:
    ollama = None


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PeoplePulse AI",
    page_icon="👥",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/employees.csv"
MODEL_PATH = "models/attrition_model.pkl"
ENCODER_PATH = "models/encoders.pkl"

data = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def encode_value(column, value):
    encoder = encoders[column]

    try:
        return encoder.transform([value])[0]
    except:
        return 0


def predict_risk(row):
    try:
        input_data = pd.DataFrame([{
            "Age": row["Age"],
            "Department": encode_value("Department", row["Department"]),
            "JobRole": encode_value("JobRole", row["JobRole"]),
            "YearsAtCompany": row["YearsAtCompany"],
            "MonthlyIncome": row["MonthlyIncome"],
            "JobSatisfaction": row["JobSatisfaction"],
            "WorkLifeBalance": row["WorkLifeBalance"],
            "PerformanceRating": row["PerformanceRating"],
            "OverTime": encode_value("OverTime", row["OverTime"]),
            "TrainingHours": row["TrainingHours"],
            "ProjectsCount": row["ProjectsCount"],
            "Workload": encode_value("Workload", row["Workload"]),
            "EngagementScore": row["EngagementScore"],
            "PromotionYears": row["PromotionYears"],
            "AbsenceDays": row["AbsenceDays"]
        }])

        probability = model.predict_proba(input_data)[0][1]

        return float(probability)

    except Exception:
        return 0.0


def risk_level(score):

    if score >= 0.70:
        return "High"

    elif score >= 0.40:
        return "Medium"

    else:
        return "Low"


def get_risk_signals(row):

    signals = []

    if row["EngagementScore"] < 50:
        signals.append("Low employee engagement")

    if row["JobSatisfaction"] <= 2:
        signals.append("Low job satisfaction")

    if row["WorkLifeBalance"] <= 2:
        signals.append("Poor work-life balance")

    if row["OverTime"] == "Yes":
        signals.append("Frequent overtime")

    if row["Workload"] == "High":
        signals.append("High workload")

    if row["AbsenceDays"] >= 8:
        signals.append("High absence days")

    if row["TrainingHours"] < 10:
        signals.append("Low training exposure")

    if row["PromotionYears"] >= 3:
        signals.append("Long time since promotion")

    if len(signals) == 0:
        signals.append("No major risk signal detected")

    return signals


def get_action_plan(row, score):

    actions = []

    if row["EngagementScore"] < 50:
        actions.append(
            "Schedule a one-to-one discussion to understand engagement concerns."
        )

    if row["OverTime"] == "Yes":
        actions.append(
            "Review workload and overtime distribution."
        )

    if row["Workload"] == "High":
        actions.append(
            "Evaluate workload and redistribute tasks if required."
        )

    if row["TrainingHours"] < 10:
        actions.append(
            "Recommend targeted learning or upskilling."
        )

    if row["PromotionYears"] >= 3:
        actions.append(
            "Review career progression and growth opportunities."
        )

    if row["JobSatisfaction"] <= 2:
        actions.append(
            "Discuss job satisfaction and workplace concerns with HR."
        )

    if len(actions) == 0:
        actions.append(
            "Continue regular employee engagement and performance monitoring."
        )

    if score >= 0.70:
        level = "High Risk"

    elif score >= 0.40:
        level = "Medium Risk"

    else:
        level = "Low Risk"

    return level, actions


# --------------------------------------------------
# LOCAL AI
# --------------------------------------------------

def local_ai(prompt):

    # Streamlit Cloud / Ollama unavailable
    if ollama is None:
        return None

    try:

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are PeoplePulse AI, an HR decision-support assistant. "
                        "Provide concise, professional and ethical HR insights. "
                        "Never recommend automatic firing or hiring. "
                        "Final decisions must remain with human HR professionals."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception:
        return None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("👥 PeoplePulse AI")

st.sidebar.caption(
    "AI Workforce Early-Warning & Decision Intelligence Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🚨 Employee Risk",
        "🤖 AI Action Planner",
        "🧠 Generative AI",
        "🎯 What-If Simulator",
        "📊 Workforce Analytics"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "AI is used for decision support. "
    "Final workforce decisions remain with human HR professionals."
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.title("👥 PeoplePulse AI")

    st.subheader(
        "AI Workforce Early-Warning & Decision Intelligence Platform"
    )

    st.write(
        "Detect workforce risks, understand why they occur, "
        "recommend actions and simulate possible interventions."
    )

    st.divider()

    # Calculate high-risk employees
    high_risk_count = sum(
        predict_risk(row) >= 0.60
        for _, row in data.iterrows()
    )

    historical_attrition = sum(
        data["Attrition"] == "Yes"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Total Employees",
        len(data)
    )

    c2.metric(
        "🚨 High-Risk Employees",
        high_risk_count
    )

    c3.metric(
        "📉 Historical Attrition",
        historical_attrition
    )

    c4.metric(
        "📊 Avg Engagement",
        round(data["EngagementScore"].mean(), 1)
    )

    st.divider()

    st.header("🎯 Selected Employee Snapshot")

    selected_id = st.selectbox(
        "Select Employee",
        data["EmployeeID"].tolist()
    )

    employee = data[
        data["EmployeeID"] == selected_id
    ].iloc[0]

    score = predict_risk(employee)

    level = risk_level(score)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Employee ID",
        employee["EmployeeID"]
    )

    c2.metric(
        "Risk Score",
        f"{score:.0%}"
    )

    c3.metric(
        "Risk Level",
        level
    )

    st.divider()

    st.subheader("🔍 Risk Signals")

    signals = get_risk_signals(employee)

    for signal in signals:
        st.write("•", signal)


# ==================================================
# EMPLOYEE RISK
# ==================================================

elif page == "🚨 Employee Risk":

    st.title("🚨 Workforce Risk Radar")

    st.write(
        "Use the ML model to identify employees who may require "
        "additional HR attention."
    )

    selected_id = st.selectbox(
        "Select Employee",
        data["EmployeeID"].tolist(),
        key="risk_employee"
    )

    employee = data[
        data["EmployeeID"] == selected_id
    ].iloc[0]

    score = predict_risk(employee)

    level = risk_level(score)

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Employee",
        employee["EmployeeID"]
    )

    c2.metric(
        "Attrition Risk",
        f"{score:.1%}"
    )

    c3.metric(
        "Risk Level",
        level
    )

    st.divider()

    st.subheader("🔎 Why is this employee at risk?")

    signals = get_risk_signals(employee)

    for signal in signals:
        st.warning(signal)

    st.divider()

    st.subheader("👤 Employee Information")

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "Department",
        employee["Department"]
    )

    info2.metric(
        "Job Role",
        employee["JobRole"]
    )

    info3.metric(
        "Engagement",
        employee["EngagementScore"]
    )

    info4.metric(
        "Performance",
        employee["PerformanceRating"]
    )

    st.subheader("📋 Employee Record")

    st.dataframe(
        pd.DataFrame([employee]),
        use_container_width=True
    )


# ==================================================
# AI ACTION PLANNER
# ==================================================

elif page == "🤖 AI Action Planner":

    st.title("🤖 AI Action Planner")

    st.write(
        "Convert workforce risk signals into practical HR actions."
    )

    selected_id = st.selectbox(
        "Select Employee",
        data["EmployeeID"].tolist(),
        key="planner_employee"
    )

    employee = data[
        data["EmployeeID"] == selected_id
    ].iloc[0]

    score = predict_risk(employee)

    level, actions = get_action_plan(
        employee,
        score
    )

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric(
        "Risk Score",
        f"{score:.1%}"
    )

    c2.metric(
        "Risk Category",
        level
    )

    st.divider()

    st.subheader("🔍 Why?")

    signals = get_risk_signals(employee)

    for signal in signals:
        st.write("•", signal)

    st.divider()

    st.subheader("💡 Recommended Actions")

    for i, action in enumerate(actions, 1):
        st.write(
            f"**{i}.** {action}"
        )

    st.divider()

    st.subheader("📅 Suggested 30-Day Plan")

    st.write("**Week 1:** Understand employee concerns")

    st.write("**Week 2:** Apply workload or learning intervention")

    st.write("**Week 3:** Review engagement and performance")

    st.write("**Week 4:** Evaluate intervention impact")

    st.divider()

    st.subheader("👤 Human-in-the-Loop")

    st.info(
        "PeoplePulse AI does not automatically fire, hire or penalize employees. "
        "HR professionals review the evidence and make the final decision."
    )

    st.checkbox(
        "HR review completed",
        key="hr_review"
    )


# ==================================================
# GENERATIVE AI
# ==================================================

elif page == "🧠 Generative AI":

    st.title("🧠 Generative AI HR Assistant")

    st.write(
        "Ask questions about workforce risk, employee engagement, "
        "HR actions and workforce insights."
    )

    st.divider()

    # Check Ollama availability
    if ollama is None:

        st.info(
            "💡 Generative AI is available when PeoplePulse AI "
            "is running locally with Ollama."
        )

        st.write(
            "The deployed Streamlit Cloud version provides the "
            "ML-based workforce analytics and decision-support features."
        )

    else:

        prompt = st.text_area(
            "Ask PeoplePulse AI",
            placeholder=(
                "Example: Why is employee E1024 at high risk "
                "and what actions can HR consider?"
            ),
            height=150
        )

        if st.button("✨ Generate AI Analysis"):

            if prompt.strip() == "":
                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "PeoplePulse AI is analysing..."
                ):

                    answer = local_ai(prompt)

                if answer:

                    st.subheader(
                        "🤖 AI Analysis"
                    )

                    st.write(answer)

                else:

                    st.warning(
                        "Local AI is currently unavailable. "
                        "Please make sure Ollama is running."
                    )

    st.divider()

    st.caption(
        "AI-generated decision support. Final workforce decisions "
        "remain with human HR professionals."
    )


# ==================================================
# WHAT-IF SIMULATOR
# ==================================================

elif page == "🎯 What-If Simulator":

    st.title("🎯 What-If Workforce Simulator")

    st.write(
        "Change employee factors and observe how the ML risk prediction changes."
    )

    selected_id = st.selectbox(
        "Select Employee",
        data["EmployeeID"].tolist(),
        key="simulation_employee"
    )

    employee = data[
        data["EmployeeID"] == selected_id
    ].iloc[0].copy()

    original_score = predict_risk(employee)

    st.divider()

    st.subheader("🎛️ Adjust Employee Factors")

    engagement = st.slider(
        "Engagement Score",
        min_value=0,
        max_value=100,
        value=int(employee["EngagementScore"])
    )

    training = st.slider(
        "Training Hours",
        min_value=0,
        max_value=50,
        value=int(employee["TrainingHours"])
    )

    workload = st.selectbox(
        "Workload",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(
            employee["Workload"]
        )
    )

    simulated_employee = employee.copy()

    simulated_employee["EngagementScore"] = engagement
    simulated_employee["TrainingHours"] = training
    simulated_employee["Workload"] = workload

    simulated_score = predict_risk(
        simulated_employee
    )

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric(
        "Original Risk",
        f"{original_score:.1%}"
    )

    c2.metric(
        "Simulated Risk",
        f"{simulated_score:.1%}"
    )

    st.divider()

    difference = simulated_score - original_score

    if difference < 0:

        st.success(
            f"Risk decreased by {abs(difference):.1%}"
        )

    elif difference > 0:

        st.error(
            f"Risk increased by {difference:.1%}"
        )

    else:

        st.info(
            "No significant change in predicted risk."
        )

    st.divider()

    st.subheader("🧑‍💼 Human Decision")

    st.info(
        "The simulator helps HR explore possible scenarios. "
        "It does not make the final workforce decision."
    )


# ==================================================
# WORKFORCE ANALYTICS
# ==================================================

elif page == "📊 Workforce Analytics":

    st.title("📊 Workforce Analytics Dashboard")

    st.write(
        "Explore workforce patterns across departments, "
        "engagement, performance and attrition."
    )

    st.divider()

    # Department distribution
    st.subheader("🏢 Employees by Department")

    department_counts = (
        data["Department"]
        .value_counts()
    )

    st.bar_chart(
        department_counts
    )

    st.divider()

    # Engagement
    st.subheader("📈 Engagement Distribution")

    st.bar_chart(
        data[
            ["EngagementScore"]
        ]
    )

    st.divider()

    # Attrition
    st.subheader("📉 Historical Attrition")

    attrition_counts = (
        data["Attrition"]
        .value_counts()
    )

    st.bar_chart(
        attrition_counts
    )

    st.divider()

    # Workload
    st.subheader("⚙️ Workload Distribution")

    workload_counts = (
        data["Workload"]
        .value_counts()
    )

    st.bar_chart(
        workload_counts
    )

    st.divider()

    st.subheader("📋 Workforce Dataset")

    st.dataframe(
        data,
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PeoplePulse AI • AI Avengers • AI-powered workforce decision support"
)
