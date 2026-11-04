import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

DATA_PATH = "HousePricePrediction.csv"
CAT_COLS = ["MSZoning", "LotConfig", "BldgType", "Exterior1st"]
NUM_COLS = ["MSSubClass", "LotArea", "OverallCond", "YearBuilt",
            "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF"]

# Simple names for the codes used in the dataset (the model still uses the codes)
SUBCLASS = {
    20: "1-story, 1946 & newer", 30: "1-story, 1945 & older",
    40: "1-story with finished attic", 45: "1½-story, unfinished upstairs",
    50: "1½-story, finished upstairs", 60: "2-story, 1946 & newer",
    70: "2-story, 1945 & older", 75: "2½-story (all ages)",
    80: "Split or multi-level", 85: "Split foyer", 90: "Duplex (all styles)",
    120: "1-story planned community, 1946 & newer",
    150: "1½-story planned community",
    160: "2-story planned community, 1946 & newer",
    180: "Planned community, multi-level",
    190: "2-family conversion",
}
ZONING = {"RL": "Residential, low density", "RM": "Residential, medium density",
          "RH": "Residential, high density", "FV": "Floating village residential",
          "C (all)": "Commercial"}
LOT = {"Inside": "Inside lot", "Corner": "Corner lot",
       "CulDSac": "Cul-de-sac (dead-end street)",
       "FR2": "Road on 2 sides", "FR3": "Road on 3 sides"}
BLDG = {"1Fam": "Single-family house", "2fmCon": "Two-family conversion",
        "Duplex": "Duplex", "TwnhsE": "Townhouse, end unit",
        "Twnhs": "Townhouse, inside unit"}
EXTERIOR = {"AsbShng": "Asbestos shingles", "AsphShn": "Asphalt shingles",
            "BrkComm": "Common brick", "BrkFace": "Brick face",
            "CBlock": "Cinder block", "CemntBd": "Cement board",
            "HdBoard": "Hardboard", "ImStucc": "Imitation stucco",
            "MetalSd": "Metal siding", "Plywood": "Plywood", "Stone": "Stone",
            "Stucco": "Stucco", "VinylSd": "Vinyl siding",
            "Wd Sdng": "Wood siding", "WdShing": "Wood shingles"}


def label(mapping):
    return lambda code: mapping.get(code, str(code))


@st.cache_data
def load_data(path_or_file):
    df = pd.read_csv(path_or_file)
    # Same cleaning as the notebook: fill NaNs with the mode, drop rows with no price
    for col in ["MSZoning", "Exterior1st", "BsmtFinSF2", "TotalBsmtSF"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=["SalePrice"])
    return df


@st.cache_resource
def train(df):
    # "Id" is just a row number, so it is left out of the features
    X = pd.get_dummies(df[NUM_COLS + CAT_COLS], drop_first=True).astype(float)
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, list(X.columns), r2


st.title("🏠 House Price Predictor")
st.caption("Linear Regression trained on the House Price Prediction dataset")

# Load data: local CSV if present, otherwise ask for an upload
if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
else:
    up = st.file_uploader("Upload HousePricePrediction.csv", type="csv")
    if up is None:
        st.info("Place HousePricePrediction.csv next to app.py, or upload it above.")
        st.stop()
    df = load_data(up)

model, columns, r2 = train(df)

st.subheader("Property details")
c1, c2 = st.columns(2)
with c1:
    ms_sub = st.selectbox("House style", sorted(df["MSSubClass"].unique()),
                          format_func=label(SUBCLASS),
                          help="Number of floors and age group of the house.")
    zoning = st.selectbox("Land use (zoning)", sorted(df["MSZoning"].unique()),
                          format_func=label(ZONING),
                          help="What the land is officially meant for.")
    lot_cfg = st.selectbox("Plot position", sorted(df["LotConfig"].unique()),
                           format_func=label(LOT),
                           help="Where the plot sits: inside, on a corner, etc.")
    bldg = st.selectbox("Type of house", sorted(df["BldgType"].unique()),
                        format_func=label(BLDG),
                        help="Single house, duplex, townhouse, etc.")
    ext = st.selectbox("Outside wall material", sorted(df["Exterior1st"].unique()),
                       format_func=label(EXTERIOR),
                       help="What the outer walls are made of.")
with c2:
    lot_area = st.number_input("Plot size (sq ft)", 1000, 250000, 9000, step=100,
                               help="Size of the whole plot of land.")
    cond = st.slider("House condition (1 = poor, 9 = excellent)", 1, 9, 5,
                     help="How well the house is kept right now.")
    built = st.number_input("Year built", 1870, 2026, 2000,
                            help="Year the house was finished.")
    remod = st.number_input("Year last renovated", 1950, 2026, 2000,
                            help="Same as year built if never renovated.")
    bsmt = st.number_input("Basement size (sq ft)", 0, 6500, 900,
                           help="Total basement area. Enter 0 if there is none.")
    bsmt2 = st.number_input("Finished basement room (sq ft)", 0, 2000, 0,
                            help="Part of the basement used as a living or play room. Enter 0 if none.")

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "MSSubClass": ms_sub, "LotArea": lot_area, "OverallCond": cond,
        "YearBuilt": built, "YearRemodAdd": remod, "BsmtFinSF2": bsmt2,
        "TotalBsmtSF": bsmt, "MSZoning": zoning, "LotConfig": lot_cfg,
        "BldgType": bldg, "Exterior1st": ext,
    }])
    row = pd.get_dummies(row).reindex(columns=columns, fill_value=0).astype(float)
    price = max(model.predict(row)[0], 0)
    st.metric("Estimated sale price", f"${price:,.0f}")
    st.caption(f"Model R² on the test set: {r2:.2f}, so treat this as a rough estimate.")
