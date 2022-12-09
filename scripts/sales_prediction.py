import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from scipy.sparse import hstack

def main():
    source = "data/vgsales.csv"                     # We're going to set the path to our data here, just for testing purposes.
    print(f"Reading training data from {source}.")  # We'll print the path as well, just to provide some transparency.
    data = pd.read_csv(source)                      # Here, we actually get around to reading the data from the source file.

    if (data.empty):
        print("Unable to read data source file. Please make sure you have the dataset in the correct position and try again.")
        return

    # Next, we select the variables we want to train on. For now, these are the titles, years, platforms, genres, and publishers
    # of each game.
    X = data[["Name", "Year", "Platform", "Genre", "Publisher"]]
    print("Selecting the data we're going to train on from our dataset...")

    # Then we select the target variable. For now, we're focusing on global scales, but we can add individual countries' sales later.
    y = data["Global_Sales"]

    # Because we want the user to be able to input arbitrary game titles, we need to do some wizardry to the titles.
    gameTitles = X["Name"].values.tolist()

    # We have to vectorize the names so that they can be used as features in our model.
    # Since they're not encodable (there aren't a set number of them) and they aren't ints (years), we need to alter them so that
    # we can pass them to our model.
    print("Vectorizing game titles...")
    vectorizer = CountVectorizer()
    vectorizer.fit(gameTitles)
    nameVectors = vectorizer.transform(gameTitles)

    # Then we can hand the rest of our data to OneHotEncoder.
    # We have to separate our other data from our title data and transform it.
    print("Encoding year, platform, genre, and publisher data...")
    encoder = OneHotEncoder(handle_unknown="ignore")
    xOther = X.drop("Name", axis=1)
    encoder.fit(xOther)
    otherVectors = encoder.transform(xOther)

    # Now we're going to concatenate both back together.
    print("Concatenating data...")
    xEncoded = hstack([nameVectors, otherVectors])

    # And now we're going to normalize the data so that our predictions can be more accurate.
    print("Normalizing data...")
    scaler = MaxAbsScaler()
    scaler.fit(xEncoded)
    xNormalized = scaler.transform(xEncoded)

    # We went with a Random Forest model because it provides more accurate predictions
    # for the kind of data we're dealing with.
    print("Training random forest regression model on our data...")
    model = RandomForestRegressor(n_estimators=10)
    xNormalized = xNormalized[:len(y)]
    model.fit(xNormalized, y)
    print("Model trained!")

    # Now we need to ask the user to input the data for the game we're going to predict the sales of.
    # We should ideally change this when we get around to creating a user interface (if we do).
    title = input("Enter the title of your game: ")
    year = int(input("Enter the year the game was released: "))
    platform = input("Enter the platform the game was released on: ")
    genre = input("Enter the genre of your game: ")
    publisher = input("Enter the publisher of your game: ")

    # Check if the input data contains any NaN values
    if any(pd.isnull([title, year, platform, genre, publisher])):
        print("One or more of the input values is invalid. Please try again.")
        return

    print("Thank you!")

    # Again, we have to do some magic to the title to transform it into usable data.
    print("Processing user inputs...")

    # We were running into a pernicious error, where OneHotEncoder would complain that
    # it was fitted to data with feature names while the data being predicted lacked
    # names, so we've added this to fix that:
    gameTitleVector = vectorizer.transform([title])

    gameOther = pd.DataFrame(data={
        "Year": [year],
        "Platform": [platform],
        "Genre": [genre],
        "Publisher": [publisher]
    })

    # And now we can transform the other data for the game via our encoder, passing the feature names as a parameter.
    gameOtherVectors = encoder.transform(gameOther)

    # Again, we have to concatenate these into a single set of data.
    gameVector = hstack([gameTitleVector, gameOtherVectors])

    # And normalize the result so that it vibes with our other data.
    gameVectorNormalized = scaler.transform(gameVector)

    # And finally we can predict our hypothetical game's sales based on the data
    # we've input. Voila!
    print("Creating sales prediction...")
    predictedSales = model.predict(gameVectorNormalized)
    print("Voila!")
    predictedSales[0] = predictedSales.round(4)
    print(f"Predicted sales: ${', '.join(map(str,predictedSales))} million.")

    # We're also going to calculate the r-squared of the model, just for kicks.
    # I wanted to see how accurate the model was.
    r2 = model.score(xNormalized, y)
    r2 = r2.round(8)
    print(f"R-squared value of the model: {r2}.")
    return


if __name__ == "__main__":
    main()