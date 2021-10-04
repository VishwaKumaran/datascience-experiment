#!/usr/bin/env python3

# Load Packages

import sklearn
import pandas

# Load Dataset
df = pd.read_csv(
	"data/dataset.csv",
	sep = ";"
)

print(df.shape)

# Clean Dataset
df = df[df["co2"].isna() == False]

df = df.drop(
	columns = ["data_ma"]
)

# Splits Values
y = df["co2"].values()

x = df.drop(
	columns = ["co2"]
)

X_train, X_test, y_train, y_test = train_test_split(
	X, 
	y, 
	test_size = 0.25, 
	random_state = 42
)

automl = AutoML(
	total_time_limit = 5*60, 
	mode = "Explain",
	random_state = 42, 
	ml_task = "regression"
)

# Fits Models
automl.fit(X_train, y_train)

# Predictions
predictions = automl.predict(X_test)

# Generate html report
automl.report()