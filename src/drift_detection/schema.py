"""Feature roles for Adult Census Income (CSV with quoted headers)."""

# Target — excluded from feature drift by default
TARGET_COL = "income"

# Numeric columns (continuous / ordinal encoded as numbers)
ADULT_NUMERIC_FEATURES = (
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
)

# Categorical columns
ADULT_CATEGORICAL_FEATURES = (
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
)
