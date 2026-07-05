import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from category_encoders import TargetEncoder
import joblib

#Load the dataset
df = pd.read_csv("data/car_price_prediction.csv")
print(df.head(10))

#Shape 
print("Shape : " , df.shape)

#columns
print (df.columns)

#Check missing values
print(df.isnull().sum())        #No missing values


#Info
print (df.info())


#Divide the independent and dependent features


df["Levy"] = df["Levy"].str.replace("-" , "0")
df["Levy"] = pd.to_numeric(df["Levy"] , errors="coerce")

#Making a new column finalprice = Price + Levy(Tax)
df["FinalPrice"] = df["Price"] + df["Price"]

print(df.columns)

X = df.drop(columns=["FinalPrice" , "Price" , "Levy" , "Model" , "Engine volume" , "Mileage" , "Doors" , "Wheel"])
y = df["FinalPrice"]


#ohe -->> [Leather interior , Fuel Type ,  Gear box type , Drive wheels ]
#Target encoding -->> [Manufacturer , Category , Color]

#Train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.25 ,random_state=42)

#One Hot Encoding
ohe = OneHotEncoder()
ohe_cols = ["Leather interior" , "Fuel type" ,  "Gear box type" , "Drive wheels"]
X_train = pd.get_dummies(X_train , columns=ohe_cols , drop_first=True , dtype=float)
X_test = pd.get_dummies(X_test , columns=ohe_cols , drop_first=True , dtype=float)


#Target encoding 
tar_enc = TargetEncoder()
X_train["Manufacturer"] = tar_enc.fit_transform(X_train["Manufacturer"] , y_train)
X_test["Manufacturer"] = tar_enc.transform(X_test["Manufacturer"])

X_train["Category"] = tar_enc.fit_transform(X_train["Category"] , y_train)
X_test["Category"] = tar_enc.transform(X_test["Category"])


X_train["Color"] = tar_enc.fit_transform(X_train["Color"] , y_train)
X_test["Color"] = tar_enc.transform(X_test["Color"])

#ALign the data 
X_train , X_test = X_train.align(X_test , join="left" , axis=1 , fill_value=0.0)

#Convert the whole data into float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

print("X_train shape : " , X_train.shape)   #(14427, 19)
print("X_test_shape : " , X_test.shape)     #(4810, 19)

# Save preprocessed data
joblib.dump(X_train, 'models/X_train.pkl')
joblib.dump(X_test, 'models/X_test.pkl')
joblib.dump(y_train, 'models/y_train.pkl')
joblib.dump(y_test, 'models/y_test.pkl')

print("All files saved successfully.")