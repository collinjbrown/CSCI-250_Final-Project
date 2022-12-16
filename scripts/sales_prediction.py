import pandas as pd

import random
import sys

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QLineEdit, QVBoxLayout, QComboBox

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.multioutput import MultiOutputRegressor
from scipy.sparse import hstack

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT

from matplotlib.figure import Figure

model = None
vectorizer = None
encoder = None
scaler = None

class MainWindow(QWidget):
    def __init__(self, graph_data):
        super().__init__()
        self.setWindowTitle("Video Game Sales Predictor")
        self.setMinimumWidth(1600)
        self.setMinimumHeight(800)

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
        platforms = graph_data["Platform"].unique()
        platforms.sort()
        for platform in platforms:
            self.platform_input.addItem(platform)

        # Create a label and combo box for the game genre.
        self.genre_label = QLabel("Enter the genre of your game:")
        self.genre_input = QComboBox()
        genres = graph_data["Genre"].unique()
        genres.sort()
        for genre in genres:
            self.genre_input.addItem(genre)
            
        # Create a label and line edit for the game publisher.
        self.publisher_label = QLabel("Enter the publisher of your game:")
        self.publisher_input = QComboBox()
        # This creates a big dropdown box of publishers.
        publishers = graph_data["Publisher"].unique()
        for publisher in publishers:
            self.publisher_input.addItem(str(publisher))
        # Publishers has issues for some reason, have to use a different method to sort
        self.publisher_input.model().sort(0)

        # Create a button to submit the user input.
        self.submit_button = QPushButton("Predict Sales")
        self.submit_button.clicked.connect(self.predict)

        # Create a label and line edit to display the global sales prediction.
        self.gb_prediction_label = QLabel("Predicted Global Sales:")
        self.gb_prediction_output = QLineEdit()
        self.gb_prediction_output.setReadOnly(True)
        
        # Create a label and line edit to display the North American sales prediction.
        self.na_prediction_label = QLabel("Predicted North American Sales:")
        self.na_prediction_output = QLineEdit()
        self.na_prediction_output.setReadOnly(True)
        
        # Create a label and line edit to display the EU sales prediction.
        self.eu_prediction_label = QLabel("Predicted EU Sales:")
        self.eu_prediction_output = QLineEdit()
        self.eu_prediction_output.setReadOnly(True)
        
        # Create a label and line edit to display the Japanese sales prediction.
        self.jp_prediction_label = QLabel("Predicted Japanese Sales:")
        self.jp_prediction_output = QLineEdit()
        self.jp_prediction_output.setReadOnly(True)

        # Create a chart to display the prediction.
        # Testing chart creation atm...
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # These are test plots, plot #3 is in def predict()
        self.graph_data = graph_data
        self.axs = self.fig.subplots(1, 5)
        self.select_genre = 'Sports'
        self.select_platform = 'Wii'
        var_graph_data = graph_data[graph_data['Platform'] == self.select_platform]
        self.axs[0].plot(var_graph_data.Year, var_graph_data.Global_Sales, 'c.', markersize = 1)
        self.axs[1].plot('Year', 'NA_Sales', 'r.', data=graph_data)
        self.axs[2].plot('Year', 'EU_Sales', 'm.', data=graph_data)
        self.axs[3].plot('Year', 'JP_Sales', 'g.', data=graph_data)
        self.axs[4].plot('Year', 'Global_Sales', 'c.', data=graph_data)

        self.axs[1].set_title("North American Sales Per Year")
        self.axs[2].set_title("EU Sales Per Year")
        self.axs[3].set_title("Japanese Sales Per Year")
        self.axs[4].set_title("Global Sales Per Year")

        self.canvas.draw()

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
        self.layout.addWidget(self.gb_prediction_label)
        self.layout.addWidget(self.gb_prediction_output)
        self.layout.addWidget(self.na_prediction_label)
        self.layout.addWidget(self.na_prediction_output)
        self.layout.addWidget(self.eu_prediction_label)
        self.layout.addWidget(self.eu_prediction_output)
        self.layout.addWidget(self.jp_prediction_label)
        self.layout.addWidget(self.jp_prediction_output)

        self.setLayout(self.layout)

        # Let user pick which axis to graph
        self.x_label_label = QLabel("Pick a category to view a chart of:") # :p
        self.x_label = QComboBox()
        self.x_label.addItem("Genre")
        self.x_label.addItem("Platform")
        self.x_label.currentIndexChanged.connect(self.update_label)

        self.x_choice = QComboBox()
        self.x_choice.currentIndexChanged.connect(self.update_choice)
        self.update_label()
        
        self.layout.addWidget(self.x_label_label)
        self.layout.addWidget(self.x_label)
        self.layout.addWidget(self.x_choice)
        
    def update_label(self):
        label = self.x_label.currentText()
        
        if label == 'Genre':
            genres = self.graph_data["Genre"].unique()
            self.x_choice.clear()
            genres.sort()
            for genre in genres:
                self.x_choice.addItem(genre)
        else:
            platforms = self.graph_data["Platform"].unique()
            self.x_choice.clear()
            platforms.sort()
            for platform in platforms:
                self.x_choice.addItem(platform)

        self.update_choice()
                
    def update_choice(self):
        self.axs[0].clear()

        label = self.x_label.currentText()
        
        new_data = ''

        self.axs[0].clear()
        if label == 'Genre':
            self.select_genre = self.x_choice.currentText()
            self.axs[0].set_title(f"{self.select_genre} Sales Per Year")
            new_data = self.graph_data[self.graph_data['Genre'] == self.select_genre]
        else:
            self.select_platform = self.x_choice.currentText()
            self.axs[0].set_title(f"{self.select_platform} Sales Per Year")
            new_data = self.graph_data[self.graph_data['Platform'] == self.select_platform]
            
        self.axs[0].plot(new_data.Year, new_data.Global_Sales, 'c.')
        self.canvas.draw()
        
    def predict(self):
        title = self.title_input.text()
        year = int(self.year_input.currentText())
        platform = self.platform_input.currentText()
        genre = self.genre_input.currentText()
        publisher = self.publisher_input.currentText()
    
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
        print(prediction)

        gb_prediction = prediction[0].round(4)
        na_prediction = prediction[1].round(4)
        eu_prediction = prediction[2].round(4)
        jp_prediction = prediction[3].round(4)

        self.gb_prediction_output.setText(str(gb_prediction))
        self.na_prediction_output.setText(str(na_prediction))
        self.eu_prediction_output.setText(str(eu_prediction))
        self.jp_prediction_output.setText(str(jp_prediction))

        new_data = {'Year': [game_other["Year"]], 'Global_Sales': [prediction]}
        
        color_set = random.choice([ 'b', 'g', 'r', 'm', 'y' ])
        style = '{}.'.format(color_set)
        
        self.axs[1].plot('Year', 'NA_Sales', style, data=new_data)
        self.axs[2].plot('Year', 'EU_Sales', style, data=new_data)
        self.axs[3].plot('Year', 'JP_Sales', style, data=new_data)
        self.axs[4].plot('Year', 'Global_Sales', style, data=new_data)
        self.canvas.draw()

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
    y = data[["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales"]]

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
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10))
    x_normalized = x_normalized[:len(y)]
    model.fit(x_normalized, y)
    print("Model trained!")

    # We're also going to calculate the r-squared of the model, just for kicks.
    # I wanted to see how accurate the model was.
    r2 = model.score(x_normalized, y)
    r2 = r2.round(8)
    print(f"R-squared value of the model: {r2}.")

    app = QApplication(sys.argv)
    window = MainWindow(data) # Took out graph_data and passed the data instead
    window.show()             # Manipulate dataframe inside MainWindow class instead
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()