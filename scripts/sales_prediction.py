import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from scipy.sparse import hstack

# First, we need to read our data from its csv file.
source = "data/vgsales.csv"
print(f"Reading training data from {source}.")
df = pd.read_csv(source)

# Next, we select the variables we want to train on. For now, these are the titles, years, platforms, genres, and publishers
# of each game.
X = df[["Name", "Year", "Platform", "Genre", "Publisher"]].rename(columns={
    "Name": "name",
    "Year": "year",
    "Platform": "platform",
    "Genre": "genre",
    "Publisher": "publisher"
})

print("Selecting the data we're going to train on from our dataset...")

# Then we select the target variable. For now, we're focusing on global scales, but we can add individual countries' sales later.
y = df["Global_Sales"]

# Because we want the user to be able to input arbitrary game titles, we need to do some wizardry to the titles.
game_titles = X["name"].values.tolist()

# We have to vectorize the names so that they can be used as features in our model.
print("Vectorizing game titles...")
vectorizer = CountVectorizer()
vectorizer.fit(game_titles)
name_vectors = vectorizer.transform(game_titles)

X_names = X.columns

# Then we can hand the rest of our data to OneHotEncoder.
# We have to separate our other data from our title data and transform it.
print("Encoding year, platform, genre, and publisher data...")
encoder = OneHotEncoder(handle_unknown="ignore")
X_other = X.drop("name", axis=1)
encoder.fit(X_other)
other_vectors = encoder.transform(X_other)

# Now we're going to concatenate both back together.
print("Concatenating data...")
X_encoded = hstack([name_vectors, other_vectors])

# And now we're going to normalize the data so that our predictions can be more accurate.
print("Normalizing data...")
scaler = MaxAbsScaler()
scaler.fit(X_encoded)
X_normalized = scaler.transform(X_encoded)

# We went with a Random Forest model because our data is skewed and abnormal and a regular
# multiple regression model wouldn't provide accurate predictions.
print("Training random forest regression model on our data...")
model = RandomForestRegressor(n_estimators=10)
X_normalized = X_normalized[:len(y)]
model.fit(X_normalized, y)
print("Model trained!")

# Now we need to ask the user to input the data for the game we're going to predict the sales of.
title = input("Enter the title of your game: ")
year = int(input("Enter the year the game was released: "))
platform = input("Enter the platform the game was released on: ")
genre = input("Enter the genre of your game: ")
publisher = input("Enter the publisher of your game: ")

# Check if the input data contains any NaN values
if any(pd.isnull([title, year, platform, genre, publisher])):
    print("One or more of the input values is invalid. Please try again.")
else:
    print("Thank you!")

    # Again, we have to do some magic to the title to transform it into usable data.
    print("Processing user inputs...")

    game_title_vector = vectorizer.transform([title])

    # We need to create a DataFrame that contains the other data for the game the user has input.
    game_other = pd.DataFrame(data={
        "year": [year],
        "platform": [platform],
        "genre": [genre],
        "publisher": [publisher]
    })

    # And now we can transform the other data for the game via our encoder, passing the feature names as a parameter.
    game_other_vectors = encoder.transform(game_other)

    # Again, we have to concatenate these into a single set of data.
    game_vector = hstack([game_title_vector, game_other_vectors])

    # And normalize the result so that it vibes with our other data.
    game_vector_normalized = scaler.transform(game_vector)

    # And finally we can predict our hypothetical game's sales based on the data
    # we've input. Voila!
    print("Creating sales prediction...")
    predicted_sales = model.predict(game_vector_normalized)
    print("Voila!")
    print(f"Predicted sales: ${', '.join(map(str,predicted_sales))} million.")
