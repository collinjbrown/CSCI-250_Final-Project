# CSCI-250_Final-Project
 The final project for CSCI 250, Fall 2022. Given a title, a year, a genre, and a publisher,
 our program will predict the sales of your hypothetical game based off of a random forest regression algorithm.


# Instructions
 - Open anaconda or conda prompt or whatever you use to run py scripts.
 - Path to the folder that contains this readme.
 - Run "python scripts/sales_prediction.py"
 - Input the requested data.
    - While you can input any title you want, the publisher and genre should be selected
    from those that are represented in the data we draw on. Ideally, we would implement
    some sort of dropdown in the interface so that the user can't choose a publisher and
    genre that isn't represented in our data.
 - Wait half a second (quite literally) for the prediction to be generated.
 - Enjoy!
 
# To Do
 - Add an interface to make interactions easier (and more constrained).
 - Add some neat features such as allowing the prediction of sales in particular countries.
 - ChatGPT has recommended the following features:
    - "Implement a recommendation system to suggest games to users based on their input data and the sales data."
    - "Allow users to specify a target sales value, and make predictions on whether a given game is likely to reach that target."
    - "Adding support for multiple languages. You could use a library like spaCy to automatically detect the language of the user's input and translate it to English before feeding it to the model. This would allow users to enter game titles, platforms, genres, and publishers in their own language, which could make the tool more user-friendly."
    - "Adding a visual representation of the model's performance. You could create a chart or graph that shows how well the model is performing on the data it's been trained on, using metrics like mean squared error and r-squared. This could help users better understand how accurate the model is, and give them more confidence in its predictions."
      - "One option is to use a library like matplotlib or seaborn to create a scatter plot of your data and show how well the model fits the data. You can also use a library like plotly to create an interactive visual that allows users to explore the model and its predictions. Additionally, you can use a library like TensorBoard to create visualizations of neural network models."

# Screenshots
![A screenshot of the command line.](https://github.com/collinjbrown/CSCI-250_Final-Project/blob/9d6301bdf271fb21a1b4e00d28da9eb55cb10596/screenshots/Capture5.PNG)