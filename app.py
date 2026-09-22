import streamlit as st
import pandas as pd
import joblib

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
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

div[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

data = pd.read_csv("data/employees.csv")

model = joblib.load(
    "models/attrition_model.pkl"
)

encoders = joblib.load(
    "models/encoders.pkl"
)


# --------------------------------------------------
# ML FUNCTIONS
# --------------------------------------------------

def predict_risk(emp):

    row = emp.drop(
        labels=["EmployeeID", "Attrition"]
    ).to_frame().T

    for col, encoder in encoders.items():
        row[col] = encoder.transform(
            row[col]
        )

    return model.predict_proba(row)[0][1]


def risk_level(p):

    if p >= 0.60:
        return "🔴 High Risk"

    elif p >= 0.30:
        return "🟠 Medium Risk"

    else:
        return "🟢 Low Risk"


# --------------------------------------------------
# RISK SIGNALS
# --------------------------------------------------

def risk_signals(emp):

    signals = []

    if emp["EngagementScore"] < 60:
        signals.append(
            "Low employee engagement"
        )

    if emp["JobSatisfaction"] <= 2:
        signals.append(
            "Low job satisfaction"
        )

    if emp["WorkLifeBalance"] <= 2:
        signals.append(
            "Poor work-life balance"
        )

    if emp["OverTime"] == "Yes":
        signals.append(
            "Frequent overtime"
        )

    if emp["Workload"] == "High":
        signals.append(
            "High workload"
        )

    if emp["TrainingHours"] < 15:
        signals.append(
            "Low recent training"
        )

    if emp["PromotionYears"] >= 4:
        signals.append(
            "Long time since promotion"
        )

    if emp["AbsenceDays"] >= 10:
        signals.append(
            "High absence days"
        )

    return signals


# --------------------------------------------------
# ACTION PLAN
# --------------------------------------------------

def action_plan(emp):

    actions = []

    if emp["EngagementScore"] < 60:
        actions.append(
            "Conduct a one-to-one engagement discussion to understand employee concerns."
        )

    if emp["Workload"] == "High":
        actions.append(
            "Review workload and redistribute tasks where possible."
        )

    if emp["OverTime"] == "Yes":
        actions.append(
            "Review overtime patterns and identify workload pressure."
        )

    if emp["TrainingHours"] < 15:
        actions.append(
            "Create a targeted learning and upskilling plan."
        )

    if emp["JobSatisfaction"] <= 2:
        actions.append(
            "Discuss the main sources of job dissatisfaction."
        )

    if emp["PromotionYears"] >= 4:
        actions.append(
            "Review career progression and internal mobility options."
        )

    if emp["AbsenceDays"] >= 10:
        actions.append(
            "Review absence patterns and workplace conditions."
        )

    return actions


# --------------------------------------------------
# LOCAL GENERATIVE AI
# --------------------------------------------------

def local_ai(prompt):

    if ollama is None:
        return None

    try:

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 120
            },
            keep_alive="10m"
        )

        return response["message"]["content"]

    except Exception as e:

        return f"Ollama Error: {e}"


# --------------------------------------------------
# AI PROMPT
# --------------------------------------------------

def ai_prompt(
    emp,
    eid,
    pct,
    signals
):

    signal_text = (
        ", ".join(signals)
        if signals
        else "None"
    )

    return f"""
You are PeoplePulse AI, an HR decision-support assistant.

Analyze this employee using ONLY these facts.

ID: {eid}
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

Give a SHORT response, maximum 120 words:

• Risk summary
• 2-3 key reasons
• 2-3 HR actions
• One 30-day step

Do not invent facts.

Do not recommend automatic firing or rejection.

HR makes the final decision.
"""


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        "## 🤖 PeoplePulse AI"
    )

    st.caption(
        "Workforce Decision Intelligence"
    )

    st.divider()

    page = st.radio(
        "Navigate to",
        [
            "🏠 Dashboard",
            "👤 Employee Risk",
            "🤖 AI Action Planner",
            "✨ Generative AI",
            "🔮 What-If Simulator",
            "📊 Workforce Analytics"
        ],
        key="main_navigation"
    )

    st.divider()

    employee_id = st.selectbox(
        "👤 Select Employee",
        data["EmployeeID"].tolist(),
        key="global_employee"
    )

    st.divider()

    st.markdown(
        "### 🎯 AI Pipeline"
    )

    st.write(
        "📊 Data → 🤖 ML Prediction → 🔎 Explain → "
        "✨ Generate → ⚡ Recommend → 👤 Human Decision"
    )

    st.divider()

    if ollama is not None:

        st.caption(
            "🦙 Local AI: Llama 3.2:3b"
        )

        st.caption(
            "No paid API required"
        )

    else:

        st.caption(
            "☁️ Cloud mode: ML + Analytics"
        )

    st.caption(
        "Final decisions remain with HR."
    )


# --------------------------------------------------
# SELECTED EMPLOYEE
# --------------------------------------------------

employee = data[
    data["EmployeeID"] == employee_id
].iloc[0]

prob = predict_risk(employee)

pct = round(
    prob * 100,
    1
)

level = risk_level(prob)

signals = risk_signals(
    employee
)

actions = action_plan(
    employee
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.title(
        "👥 PeoplePulse AI"
    )

    st.subheader(
        "AI Workforce Early-Warning & Decision Intelligence Platform"
    )

    st.write(
        "Detect workforce risks, understand why they are happening, "
        "generate HR recommendations, and simulate possible interventions."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    high_risk_count = sum(
        predict_risk(row) >= 0.60
        for _, row in data.iterrows()
    )

    c1.metric(
        "👥 Total Employees",
        len(data)
    )

    c2.metric(
        "🚨 High-Risk Employees",
        high_risk_count
    )

    c3.metric(
        "📊 Historical Attrition",
        int(
            (data["Attrition"] == "Yes").sum()
        )
    )

    c4.metric(
        "💡 Avg Engagement",
        round(
            data["EngagementScore"].mean(),
            1
        )
    )

    st.divider()

    st.header(
        "🎯 Selected Employee Snapshot"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Employee",
        employee_id
    )

    c2.metric(
        "ML Attrition Risk",
        f"{pct}%"
    )

    c3.metric(
        "Risk Level",
        level
    )

    st.info(
        "Use the sidebar to open each working module."
    )

    st.subheader(
        "🚨 Current Risk Signals"
    )

    if signals:

        for x in signals:

            st.warning(
                "⚠️ " + x
            )

    else:

        st.success(
            "✅ No major rule-based risk signals detected."
        )


# ==================================================
# EMPLOYEE RISK
# ==================================================

elif page == "👤 Employee Risk":

    st.title(
        "👤 Employee Risk Analysis"
    )

    st.write(
        "ML-based attrition risk with explainable workforce signals."
    )

    st.subheader(
        f"Employee Profile — {employee_id}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.write(
            "**Department:**",
            employee["Department"]
        )

        st.write(
            "**Job Role:**",
            employee["JobRole"]
        )

        st.write(
            "**Years at Company:**",
            employee["YearsAtCompany"]
        )

    with c2:

        st.write(
            "**Monthly Income:** ₹",
            employee["MonthlyIncome"]
        )

        st.write(
            "**Job Satisfaction:**",
            employee["JobSatisfaction"],
            "/ 4"
        )

        st.write(
            "**Work-Life Balance:**",
            employee["WorkLifeBalance"],
            "/ 4"
        )

    with c3:

        st.write(
            "**Performance Rating:**",
            employee["PerformanceRating"]
        )

        st.write(
            "**Engagement Score:**",
            employee["EngagementScore"],
            "/ 100"
        )

        st.write(
            "**Workload:**",
            employee["Workload"]
        )

    st.divider()

    st.subheader(
        "🚨 Risk Signals"
    )

    if signals:

        for x in signals:

            st.warning(
                "⚠️ " + x
            )

    else:

        st.success(
            "✅ No major workforce risk signals detected."
        )

    st.divider()

    st.subheader(
        "🤖 AI Attrition Risk Prediction"
    )

    if prob >= 0.60:

        st.error(
            f"🔴 High Attrition Risk — Model Probability: {pct}%"
        )

    elif prob >= 0.30:

        st.warning(
            f"🟠 Medium Attrition Risk — Model Probability: {pct}%"
        )

    else:

        st.success(
            f"🟢 Low Attrition Risk — Model Probability: {pct}%"
        )

    st.progress(
        min(
            max(prob, 0),
            1
        )
    )

    st.divider()

    st.subheader(
        "📋 Employee Data"
    )

    st.dataframe(
        employee.to_frame().T,
        use_container_width=True
    )


# ==================================================
# AI ACTION PLANNER
# ==================================================

elif page == "🤖 AI Action Planner":

    st.title(
        "🤖 AI Action Planner"
    )

    st.write(
        "Convert detected workforce risks into a structured HR action plan."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Employee",
        employee_id
    )

    c2.metric(
        "ML Risk",
        f"{pct}%"
    )

    c3.metric(
        "Risk Level",
        level
    )

    st.divider()

    st.subheader(
        "🧠 Why is this employee at risk?"
    )

    if signals:

        for x in signals:

            st.warning(
                "⚠️ " + x
            )

    else:

        st.success(
            "No major risk drivers detected."
        )

    st.subheader(
        "⚡ Recommended Actions"
    )

    if actions:

        for i, x in enumerate(
            actions,
            1
        ):

            st.info(
                f"**Action {i}:** {x}"
            )

    else:

        st.success(
            "No immediate intervention required."
        )

    st.subheader(
        "📅 30-Day Intervention Plan"
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            "### Week 1"
        )

        st.write(
            "🔹 HR / Manager check-in"
        )

        st.write(
            "🔹 Understand employee concerns"
        )

        st.write(
            "🔹 Review workload"
        )

    with b:

        st.markdown(
            "### Week 2–3"
        )

        st.write(
            "🔹 Apply selected intervention"
        )

        st.write(
            "🔹 Provide training/support"
        )

        st.write(
            "🔹 Monitor engagement"
        )

    with c:

        st.markdown(
            "### Week 4"
        )

        st.write(
            "🔹 Reassess employee risk"
        )

        st.write(
            "🔹 Compare new scenario"
        )

        st.write(
            "🔹 HR review"
        )

    st.divider()

    st.subheader(
        "👤 Human-in-the-Loop"
    )

    decision = st.selectbox(
        "HR Review Status",
        [
            "Select status",
            "Needs HR Review",
            "Under Manager Review",
            "Intervention Planned",
            "Continue Monitoring"
        ],
        key="planner_hr_status"
    )

    if decision != "Select status":

        st.success(
            f"✅ HR Status: {decision}"
        )

    st.caption(
        "PeoplePulse AI provides decision support. "
        "Final workforce decisions remain with human HR professionals."
    )


# ==================================================
# GENERATIVE AI
# ==================================================

elif page == "✨ Generative AI":

    st.title(
        "✨ Generative AI HR Insight"
    )

    if ollama is None:

        st.info(
            "☁️ Generative AI is available when PeoplePulse AI "
            "is running locally with Ollama. The deployed cloud "
            "version continues to provide ML-based workforce analytics."
        )

        st.write(
            "Your Dashboard, Employee Risk, AI Action Planner, "
            "What-If Simulator and Workforce Analytics remain available."
        )

    else:

        st.write(
            "Generate a natural-language HR analysis using your local Llama model."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Employee",
            employee_id
        )

        c2.metric(
            "ML Risk",
            f"{pct}%"
        )

        c3.metric(
            "Risk Level",
            level
        )

        st.info(
            "🦙 Llama 3.2:3b is running locally through Ollama. "
            "Fast demo mode: concise responses."
        )

        if signals:

            st.subheader(
                "🔎 Input Risk Signals"
            )

            for x in signals:

                st.warning(
                    "⚠️ " + x
                )

        if st.button(
            "✨ Generate AI HR Insight",
            type="primary",
            key="generate_ai_insight"
        ):

            with st.spinner(
                "🦙 Generating a concise AI insight..."
            ):

                result = local_ai(
                    ai_prompt(
                        employee,
                        employee_id,
                        pct,
                        signals
                    )
                )

            if result:

                st.markdown(
                    "### 🧠 AI-Generated HR Analysis"
                )

                st.markdown(
                    result
                )

            else:

                st.warning(
                    "Local AI is currently unavailable. "
                    "Please make sure Ollama is running."
                )

    st.caption(
        "AI-generated decision support. Final workforce decisions "
        "remain with human HR professionals."
    )


# ==================================================
# WHAT-IF SIMULATOR
# ==================================================

elif page == "🔮 What-If Simulator":

    st.title(
        "🔮 What-If Workforce Simulator"
    )

    st.write(
        "Change workforce conditions and run the same ML model again "
        "to compare predicted risk."
    )

    st.info(
        "💡 Try changing engagement, training hours or workload."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        eng = st.slider(
            "Engagement Score",
            0,
            100,
            int(
                employee["EngagementScore"]
            ),
            key="sim_engagement"
        )

    with c2:

        training = st.slider(
            "Training Hours",
            0,
            60,
            int(
                employee["TrainingHours"]
            ),
            key="sim_training"
        )

    with c3:

        opts = [
            "Low",
            "Medium",
            "High"
        ]

        workload = st.selectbox(
            "Workload",
            opts,
            index=opts.index(
                employee["Workload"]
            ),
            key="sim_workload"
        )

    sim = employee.copy()

    sim["EngagementScore"] = eng

    sim["TrainingHours"] = training

    sim["Workload"] = workload

    sim_prob = predict_risk(
        sim
    )

    sim_pct = round(
        sim_prob * 100,
        1
    )

    st.divider()

    st.subheader(
        "📊 Current vs What-If Scenario"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current ML Risk",
        f"{pct}%"
    )

    c2.metric(
        "Simulated ML Risk",
        f"{sim_pct}%",
        delta=f"{sim_pct - pct:.1f}%"
    )

    diff = round(
        sim_pct - pct,
        1
    )

    with c3:

        if diff < 0:

            st.success(
                f"📉 Risk Reduced by {abs(diff)}%"
            )

        elif diff > 0:

            st.error(
                f"📈 Risk Increased by {diff}%"
            )

        else:

            st.info(
                "➡️ Risk Unchanged"
            )

    st.subheader(
        "🧠 Scenario Details"
    )

    c1, c2, c3 = st.columns(3)

    c1.write(
        f"**Engagement:** "
        f"{employee['EngagementScore']} → {eng}"
    )

    c2.write(
        f"**Training Hours:** "
        f"{employee['TrainingHours']} → {training}"
    )

    c3.write(
        f"**Workload:** "
        f"{employee['Workload']} → {workload}"
    )

    st.subheader(
        "💡 AI Interpretation"
    )

    if sim_pct < pct:

        st.success(
            "The simulated intervention reduced the model's "
            "predicted attrition risk. HR can review this scenario "
            "before considering an actual intervention."
        )

    elif sim_pct > pct:

        st.warning(
            "The simulated scenario increased the model's "
            "predicted attrition risk. HR can explore alternative interventions."
        )

    else:

        st.info(
            "The simulated changes did not change the model's "
            "predicted attrition risk."
        )

    st.caption(
        "Simulation is a model scenario, not a guarantee of a real-world outcome."
    )


# ==================================================
# WORKFORCE ANALYTICS
# ==================================================

elif page == "📊 Workforce Analytics":

    st.title(
        "📊 Workforce Analytics Dashboard"
    )

    st.write(
        "Explore workforce patterns across departments, engagement, "
        "performance and historical attrition."
    )

    summary = (
        data.groupby("Department")
        .agg(
            Employees=("EmployeeID", "count"),
            Average_Engagement=(
                "EngagementScore",
                "mean"
            ),
            Average_Performance=(
                "PerformanceRating",
                "mean"
            ),
            Attrition_Count=(
                "Attrition",
                lambda x: (
                    x == "Yes"
                ).sum()
            )
        )
        .reset_index()
    )

    summary["Attrition_Rate"] = (
        summary["Attrition_Count"]
        / summary["Employees"]
        * 100
    ).round(1)

    summary["Average_Engagement"] = (
        summary["Average_Engagement"]
        .round(1)
    )

    summary["Average_Performance"] = (
        summary["Average_Performance"]
        .round(2)
    )

    st.subheader(
        "🏢 Employees by Department"
    )

    st.bar_chart(
        summary.set_index(
            "Department"
        )["Employees"]
    )

    st.subheader(
        "📈 Average Engagement by Department"
    )

    st.bar_chart(
        summary.set_index(
            "Department"
        )["Average_Engagement"]
    )

    st.subheader(
        "⚠️ Historical Attrition Rate by Department"
    )

    st.bar_chart(
        summary.set_index(
            "Department"
        )["Attrition_Rate"]
    )

    st.subheader(
        "🎯 Engagement vs Performance"
    )

    st.scatter_chart(
        data[
            [
                "EngagementScore",
                "PerformanceRating"
            ]
        ],
        x="EngagementScore",
        y="PerformanceRating"
    )

    st.subheader(
        "📚 Training Hours vs Engagement"
    )

    st.scatter_chart(
        data[
            [
                "TrainingHours",
                "EngagementScore"
            ]
        ],
        x="TrainingHours",
        y="EngagementScore"
    )

    st.subheader(
        "📋 Department Intelligence"
    )

    st.dataframe(
        summary,
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PeoplePulse AI • AI Avengers • AI-powered workforce decision intelligence"
)

st.caption(
    "Prototype uses synthetic employee data for demonstration purposes."
)
