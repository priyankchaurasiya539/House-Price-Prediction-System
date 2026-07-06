import pandas as pd 
from sklearn.model_selection import train_test_split
import joblib

#Load the dataset
df = pd.read_csv("data/data (1).csv")
print(df.head(10))

#Shape 
print("Shape : " , df.shape)

#Check missing values
print(df.isnull().sum())        #No missing values

#Info
print (df.info())

#Divide the independent and dependent features

print("Length of original columns : " , len(df.columns))

X = df.drop(columns=["date" , "price" ,  "sqft_living" , "sqft_lot" , "sqft_above" , "sqft_basement" , "yr_built" , "yr_renovated" , "street" , "city" , "country" , "statezip"])
y = df["price"]

print("After feature engineering : " , len(X.columns))

print(X.info())

#Train_test_split

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.25 , random_state=42)

#Converting whole data into float value

X_train = X_train.astype(float)
X_test = X_test.astype(float)

#Saving the files in models
joblib.dump(X_train, 'models/X_train.pkl')
joblib.dump(X_test, 'models/X_test.pkl')
joblib.dump(y_train, 'models/y_train.pkl')
joblib.dump(y_test, 'models/y_test.pkl')

print("All files saved successfully.")


# Load data
# X_train = joblib.load('models/X_train.pkl')
# X_test = joblib.load('models/X_test.pkl')
# y_train = joblib.load('models/y_train.pkl')
# y_test = joblib.load('models/y_test.pkl')

# print("✅ Data Loaded Successfully!")
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)

