import pandas as pd

import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QLineEdit, QVBoxLayout, QComboBox
from PyQt5.QtChart import QChart, QLineSeries, QCategoryAxis, QChartView

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from scipy.sparse import hstack

model = None
vectorizer = None
encoder = None
scaler = None

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Game Sales Predictor")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        # Create a layout for the window.
        self.layout = QVBoxLayout()

        # Create a label and line edit for the game title.
        self.title_label = QLabel("Enter the title of your game:")
        self.title_input = QLineEdit()

        # Create a label and combo box for the game year.
        self.year_label = QLabel("Enter the year the game was released:")
        self.year_input = QComboBox()
        for year in range(1980, 2023):
            self.year_input.addItem(str(year))

        # Create a label and combo box for the game platform.
        self.platform_label = QLabel("Enter the platform the game was released on:")
        self.platform_input = QComboBox()
        platforms = ["Wii", "WiiU", "NES", "SNES", "N64", "GB", "GC", "DS", "3DS", "XB", "X360", "XOne", "PS", "PSP", "PS2", "PS3", "PS4", "PC", "Atari"]
        for platform in platforms:
            self.platform_input.addItem(platform)

        # Create a label and combo box for the game genre.
        self.genre_label = QLabel("Enter the genre of your game:")
        self.genre_input = QComboBox()
        genres = ["Sports", "Platform", "Racing", "Role-Playing", "Shooter", "Simulation", "Action", "Puzzle", "Fighting", "Misc"]
        for genre in genres:
            self.genre_input.addItem(genre)

        # Create a label and line edit for the game publisher.
        self.publisher_label = QLabel("Enter the publisher of your game:")
        self.publisher_input = QLineEdit()

        # Create a button to submit the user input.
        self.submit_button = QPushButton("Predict Sales")
        self.submit_button.clicked.connect(self.predict)

        # Create a label and line edit to display the prediction.
        self.prediction_label = QLabel("Predicted Global Sales:")
        self.prediction_output = QLineEdit()
        self.prediction_output.setReadOnly(True)
        
        # # Create a chart to display the prediction.
        # self.chart = QChart()
        # self.chart.setAnimationOptions(QChart.AllAnimations)
        # self.series = QLineSeries()
        # self.chart.addSeries(self.series)
        # self.chart.createDefaultAxes()
        # self.chart.axisX().setRange(1980, 2020)
        # self.chart.axisY().setRange(0, 50)
        # self.chart.axisX().setLabelsVisible(False)
        # self.chart.axisY().setLabelsVisible(False)
        # self.chart.legend().hide()

        # self.chart_view = QChartView(self.chart)

        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.title_input)
        self.layout.addWidget(self.year_label)
        self.layout.addWidget(self.year_input)
        self.layout.addWidget(self.platform_label)
        self.layout.addWidget(self.platform_input)
        self.layout.addWidget(self.genre_label)
        self.layout.addWidget(self.genre_input)
        self.layout.addWidget(self.publisher_label)
        self.layout.addWidget(self.publisher_input)
        self.layout.addWidget(self.submit_button)
        self.layout.addWidget(self.prediction_label)
        self.layout.addWidget(self.prediction_output)
        # self.layout.addWidget(self.chart_view)

        self.setLayout(self.layout)

    # def update_chart(self, x_values, y_values, x_values_line, y_values_line):
    #     self.series.clear()
    #     self.series.append(x_values, y_values)
    #     self.series.setMarkerSize(8)

    #     self.line_series = QLineSeries()
    #     self.line_series.append(x_values_line, y_values_line)
    #     self.chart.addSeries(self.line_series)

    #     self.chart.axisX().setRange(min(x_values), max(x_values))
    #     self.chart.axisY().setRange(min(y_values), max(y_values))

    def predict(self):
        title = self.title_input.text()
        year = int(self.year_input.currentText())
        platform = self.platform_input.currentText()
        genre = self.genre_input.currentText()
        publisher = self.publisher_input.text()
    
        # Check if the input data contains any NaN values.
        if any(pd.isnull([title, year, platform, genre, publisher])):
            print("One or more of the input values is invalid. Please try again.")
            return

        game_title_vector = vectorizer.transform([title])

        game_other = pd.DataFrame(data={
            "Year": [year],
            "Platform": [platform],
            "Genre": [genre],
            "Publisher": [publisher]
        })        

        game_other_vectors = encoder.transform(game_other)
        game_vector = hstack([game_title_vector, game_other_vectors])
        game_vector_normalized = scaler.transform(game_vector)

        prediction = model.predict(game_vector_normalized)[0]

        prediction = prediction.round(4)

        self.prediction_output.setText(str(prediction))

        # self.series.append(year, prediction)

        # self.update_chart()

def main():
    global model, vectorizer, encoder, scaler

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
    print(X)

    # Then we select the target variable. For now, we're focusing on global scales, but we can add individual countries' sales later.
    y = data["Global_Sales"]

    # Because we want the user to be able to input arbitrary game titles, we need to do some wizardry to the titles.
    game_titles = X["Name"].values.tolist()

    # We have to vectorize the names so that they can be used as features in our model.
    # Since they're not encodable (there aren't a set number of them) and they aren't ints (years), we need to alter them so that
    # we can pass them to our model.
    print("Vectorizing game titles...")
    vectorizer = CountVectorizer()
    vectorizer.fit(game_titles)
    name_vectors = vectorizer.transform(game_titles)

    # Then we can hand the rest of our data to OneHotEncoder.
    # We have to separate our other data from our title data and transform it.
    print("Encoding year, platform, genre, and publisher data...")
    encoder = OneHotEncoder(handle_unknown="ignore")
    x_other = X.drop("Name", axis=1)
    encoder.fit(x_other)
    other_vectors = encoder.transform(x_other)

    # Now we're going to concatenate both back together.
    print("Concatenating data...")
    x_encoded = hstack([name_vectors, other_vectors])

    # And now we're going to normalize the data so that our predictions can be more accurate.
    print("Normalizing data...")
    scaler = MaxAbsScaler()
    scaler.fit(x_encoded)
    x_normalized = scaler.transform(x_encoded)
    
    # We went with a Random Forest model because it provides more accurate predictions
    # for the kind of data we're dealing with.
    print("Training random forest regression model on our data...")
    model = RandomForestRegressor(n_estimators=10)
    x_normalized = x_normalized[:len(y)]
    model.fit(x_normalized, y)
    print("Model trained!")
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    ## Again, we have to do some magic to the title to transform it into usable data.
    #print("Processing user inputs...")

    ## We were running into a pernicious error, where OneHotEncoder would complain that
    ## it was fitted to data with feature names while the data being predicted lacked
    ## names, so we've added this to fix that:
    #game_title_vector = vectorizer.transform([title])

    #gameOther = pd.DataFrame(data={
    #    "Year": [year],
    #    "Platform": [platform],
    #    "Genre": [genre],
    #    "Publisher": [publisher]
    #})

    ## And now we can transform the other data for the game via our encoder, passing the feature names as a parameter.
    #gameOtherVectors = encoder.transform(gameOther)

    ## Again, we have to concatenate these into a single set of data.
    #gameVector = hstack([gameTitleVector, gameOtherVectors])

    ## And normalize the result so that it vibes with our other data.
    #gameVectorNormalized = scaler.transform(gameVector)

    ## And finally we can predict our hypothetical game's sales based on the data
    ## we've input. Voila!
    #print("Creating sales prediction...")
    #predictedSales = model.predict(gameVectorNormalized)
    #print("Voila!")
    #predictedSales[0] = predictedSales.round(4)
    #print(f"Predicted sales: ${', '.join(map(str,predictedSales))} million.")

    ## We're also going to calculate the r-squared of the model, just for kicks.
    ## I wanted to see how accurate the model was.
    #r2 = model.score(xNormalized, y)
    #r2 = r2.round(8)
    #print(f"R-squared value of the model: {r2}.")

if __name__ == "__main__":
    main()