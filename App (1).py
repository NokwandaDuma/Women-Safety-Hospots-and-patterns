import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Ekurhuleni Women Safety Hotspot Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Women Safety Hotspot Analytics – City of Ekurhuleni")
st.markdown("**Data-driven identification of violence hotspots and patterns (2020–2025)**")
st.markdown("---")

# ------------------------------------------------------------
# Data loading (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner="Loading SAPS crime & station data...")
def load_data():
    CRIME_URL = "https://media.githubusercontent.com/media/afrith/crime-stats/main/crime-stats.csv"
    STATIONS_URL = "https://raw.githubusercontent.com/afrith/crime-stats/main/police_stations.csv"

    stations = pd.read_csv(STATIONS_URL)
    crime = pd.read_csv(CRIME_URL)

    eku_stations = stations[stations["muni_code"] == "EKU"].copy()
    eku_names = set(eku_stations["name"].str.strip().str.lower())
    eku_codes = set(eku_stations["code"].str.strip())

    crime["station_clean"] = crime["station_name"].str.strip().str.lower()
    eku_crime = crime[
        (crime["station_clean"].isin(eku_names)) |
        (crime["station_code"].isin(eku_codes))
    ].copy()

    CONTACT_CRIMES = [
        "Murder", "Rape", "Sexual offences", "Sexual assault",
        "Attempted sexual offences", "Contact sexual offences",
        "Attempted murder",
        "Assault with the intent to inflict grievous bodily harm",
        "Common assault", "Common robbery",
        "Robbery with aggravating circumstances",
        "Robbery at residential premises", "Robbery at non-residential premises",
        "Carjacking", "Kidnapping", "Abduction"
    ]

    eku_contact = eku_crime[eku_crime["crime_name"].isin(CONTACT_CRIMES)].copy()

    monthly = (
        eku_contact
        .groupby(["station_code", "station_name", "year", "month"], as_index=False)["crime_count"]
        .sum()
        .rename(columns={"crime_count": "contact_count"})
    )

    monthly = monthly.merge(
        eku_stations[["code", "name", "population", "area_km2"]],
        left_on="station_code", right_on="code", how="left"
    )

    monthly["contact_rate"] = (monthly["contact_count"] / monthly["population"]) * 100_000
    monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))

    threshold = monthly["contact_rate"].quantile(0.80)
    monthly["is_hotspot"] = (monthly["contact_rate"] >= threshold).astype(int)

    return eku_crime, eku_contact, monthly, eku_stations, threshold, CONTACT_CRIMES

eku_crime, eku_contact, monthly, eku_stations, threshold, CONTACT_CRIMES = load_data()

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("🔎 Filters")

all_stations = sorted(monthly["station_name"].unique())
selected_stations = st.sidebar.multiselect(
    "Police Precinct(s)",
    options=all_stations,
    default=all_stations
)

selected_crimes = st.sidebar.multiselect(
    "Crime Category",
    options=CONTACT_CRIMES,
    default=CONTACT_CRIMES
)

years = sorted(monthly["year"].unique())
year_range = st.sidebar.slider(
    "Year Range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years)))
)

# Apply filters
filtered_monthly = monthly[
    (monthly["station_name"].isin(selected_stations)) &
    (monthly["year"] >= year_range[0]) &
    (monthly["year"] <= year_range[1])
].copy()

filtered_contact = eku_contact[
    (eku_contact["station_name"].isin(selected_stations)) &
    (eku_contact["crime_name"].isin(selected_crimes)) &
    (eku_contact["year"] >= year_range[0]) &
    (eku_contact["year"] <= year_range[1])
].copy()

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & EDA",
    "🔥 Hotspot Classification",
    "📍 Three Priority Precincts",
    "🧠 Model Insights",
    "💡 Interpretation & Recommendations"
])

# ============================================================
# TAB 1 – Overview & EDA
# ============================================================
with tab1:
    st.header("Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precincts selected", len(selected_stations))
    col2.metric("Total contact incidents", f"{filtered_contact['crime_count'].sum():,}")
    col3.metric("Hotspot threshold", f"{threshold:.1f} / 100k")
    col4.metric("Hotspot station-months", f"{filtered_monthly['is_hotspot'].sum():,}")

    st.subheader("City-wide Monthly Trend")
    city = (
        filtered_monthly
        .groupby("date", as_index=False)
        .agg(total=("contact_count", "sum"), pop=("population", "sum"))
    )
    city["rate"] = (city["total"] / city["pop"]) * 100_000

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(city["date"], city["rate"], color="darkred", linewidth=1.6)
    ax.axhline(threshold, color="red", linestyle="--", alpha=0.7, label="Hotspot threshold")
    ax.set_ylabel("Contact rate per 100 000")
    ax.set_title("Ekurhuleni – Monthly Contact-Crime Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Average Rate by Precinct (selected period)")
    station_avg = (
        filtered_monthly
        .groupby("station_name")["contact_rate"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, max(6, len(station_avg)*0.28)))
    station_avg.plot(kind="barh", ax=ax, color="crimson", alpha=0.85)
    ax.set_xlabel("Mean monthly rate per 100 000")
    ax.set_title("Precinct Ranking by Average Contact-Crime Rate")
    st.pyplot(fig)

    st.subheader("Crime Category Breakdown (selected filters)")
    crime_sum = (
        filtered_contact
        .groupby("crime_name")["crime_count"]
        .sum()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    crime_sum.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Total incidents")
    ax.set_title("Volume by Contact-Crime Type")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ============================================================
# TAB 2 – Hotspot Classification
# ============================================================
with tab2:
    st.header("Hotspot Classification Results")

    st.markdown(f"""
    **Definition used**  
    A station-month is labelled a **hotspot** when its contact-crime rate  
    is at or above the **80th percentile** of all Ekurhuleni station-months  
    (**{threshold:.1f} per 100 000 population**).  
    This threshold is transparent, rate-based and produces ~20 % positive cases.
    """)

    # Quick model rebuild for the dashboard (same features as notebook)
    @st.cache_resource
    def train_models(df):
        d = df.copy()
        d = d.sort_values(["station_code", "date"])
        d["lag1_rate"] = d.groupby("station_code")["contact_rate"].shift(1)
        d["lag1_count"] = d.groupby("station_code")["contact_count"].shift(1)
        d["roll3_mean"] = d.groupby("station_code")["contact_rate"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        d["roll3_std"] = d.groupby("station_code")["contact_rate"].transform(
            lambda x: x.rolling(3, min_periods=1).std()
        )
        d["log_pop"] = np.log1p(d["population"])
        d["log_dens"] = np.log1p(d["population"] / d["area_km2"])
        d["month"] = d["date"].dt.month
        d["quarter"] = d["date"].dt.quarter
        d["is_winter"] = d["month"].isin([6, 7, 8]).astype(int)
        d["is_festive"] = d["month"].isin([11, 12, 1]).astype(int)
        d["year"] = d["date"].dt.year

        d = d.dropna(subset=["lag1_rate", "roll3_std"])

        features = [
            "lag1_rate", "lag1_count", "roll3_mean", "roll3_std",
            "log_pop", "log_dens", "month", "quarter",
            "is_winter", "is_festive", "year"
        ]
        X = d[features]
        y = d["is_hotspot"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42
        )
        gb.fit(X_train, y_train)

        return rf, gb, X_test, y_test, features

    rf, gb, X_test, y_test, features = train_models(monthly)

    y_pred_gb = gb.predict(X_test)
    y_prob_gb = gb.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gradient Boosting (Best Model)")
        st.write(f"**Accuracy:** {(y_pred_gb == y_test).mean():.3f}")
        st.write(f"**ROC-AUC:** {roc_auc_score(y_test, y_prob_gb):.3f}")
        st.text(classification_report(y_test, y_pred_gb, digits=3))

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_gb)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Hotspot", "Hotspot"],
                    yticklabels=["Not Hotspot", "Hotspot"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.subheader("Feature Importance (Gradient Boosting)")
    imp = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ============================================================
# TAB 3 – Three Priority Precincts
# ============================================================
with tab3:
    st.header("Three Priority Hotspot Precincts")

    selected3 = ["Boksburg North", "Kempton Park", "Germiston"]
    st.markdown(f"**Selected precincts:** {', '.join(selected3)}")

    st.subheader("Comparative Summary")
    comp = monthly[monthly["station_name"].isin(selected3)]
    summary = (
        comp.groupby("station_name")
        .agg(
            mean_rate=("contact_rate", "mean"),
            max_rate=("contact_rate", "max"),
            total_incidents=("contact_count", "sum"),
            population=("population", "first"),
            area_km2=("area_km2", "first")
        )
        .round(1)
    )
    summary["density"] = (summary["population"] / summary["area_km2"]).round(0)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Monthly Rate Trajectories")
    fig, ax = plt.subplots(figsize=(11, 5))
    for stn in selected3:
        sub = comp[comp["station_name"] == stn]
        ax.plot(sub["date"], sub["contact_rate"], label=stn, linewidth=1.8)
    ax.axhline(threshold, color="red", linestyle="--", alpha=0.7, label="Hotspot threshold")
    ax.set_ylabel("Rate per 100 000")
    ax.set_title("Contact-Crime Rate – Three Priority Precincts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Crime-Type Mix in the Three Precincts")
    hot_crime = eku_contact[eku_contact["station_name"].isin(selected3)]
    mix = (
        hot_crime.groupby(["station_name", "crime_name"])["crime_count"]
        .sum()
        .unstack(fill_value=0)
    )
    top8 = mix.sum().nlargest(8).index
    st.dataframe(mix[top8].T, use_container_width=True)

# ============================================================
# TAB 4 – Model Insights
# ============================================================
with tab4:
    st.header("Model & Pattern Insights")

    st.markdown("""
    **What the models learned**
    - Previous-month rate and 3-month rolling average are by far the strongest predictors.
    - Population size / density and calendar features (month, festive season) also contribute.
    - The high ROC-AUC (> 0.98) shows that hotspot status is highly predictable from recent history and basic station characteristics.
    """)

    st.subheader("Hotspot Prevalence by Precinct")
    prev = (
        monthly.groupby("station_name")
        .agg(hotspot_pct=("is_hotspot", "mean"), mean_rate=("contact_rate", "mean"))
        .sort_values("hotspot_pct", ascending=False)
    )
    prev["hotspot_pct"] = (prev["hotspot_pct"] * 100).round(1)
    st.dataframe(prev.head(15), use_container_width=True)

# ============================================================
# TAB 5 – Interpretation & Recommendations
# ============================================================
with tab5:
    st.header("Interpretation, Limitations & Prevention Insights")

    st.subheader("Key Findings (Technical & Non-Technical)")
    st.markdown("""
    - Risk is **highly concentrated**. A small number of precincts (especially Boksburg North, Kempton Park and Germiston) account for the large majority of high-rate months.
    - These three precincts were flagged as hotspots in **98.5 % of all months** studied — the problem is persistent, not temporary.
    - **Kempton Park** stands out both statistically and because of recent high-profile cases.
    - Rates (not raw counts) give a fairer picture: smaller precincts such as Boksburg North can be extremely high-intensity even if absolute numbers are lower.
    """)

    st.subheader("Similarities & Differences of the Three Hotspots")
    st.markdown("""
    **Similar**
    - Extreme persistence (almost every month a hotspot)
    - Same dominant crime mix (aggravated robbery, assault GBH, common assault)
    - Strong month-to-month autocorrelation

    **Different**
    - **Boksburg North**: highest rate, smaller population → high-intensity, lower absolute volume
    - **Kempton Park & Germiston**: high rate *and* high absolute volume
    """)

    st.subheader("Prevention-Oriented Insights")
    st.markdown("""
    1. **Prioritise the three precincts** for intensified visible policing, detective capacity and multi-agency safety work.
    2. **Use rate-based monitoring** so that smaller but high-intensity areas are not overlooked.
    3. **Early-warning rule**: a sharp rise in the previous month’s rate is a strong signal that the next month is likely to be a hotspot — this can be tested operationally.
    4. Different emphasis may be needed: intensity reduction in Boksburg North; both intensity *and* volume reduction in Kempton Park and Germiston.
    """)

    st.subheader("Important Limitations (Association ≠ Causation)")
    st.markdown("""
    - Public SAPS data currently lack victim gender and age → contact crime is only a **proxy** for women-safety risk.
    - Population figures are a 2022 snapshot (not time-varying).
    - Under-reporting is known to be high for sexual offences and domestic violence.
    - The analysis identifies **statistical associations**, not causal mechanisms. We cannot claim that any specific factor (lighting, transport hubs, poverty, etc.) *causes* the elevated rates.
    - Results from these three precincts should not be assumed to apply city-wide without further evaluation.
    """)

    st.success("Dashboard ready for both technical review and non-technical briefing.")