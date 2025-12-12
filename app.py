# Description: Flask-based web application for visualizing Spotify music data. Users can explore genres,
# select moods (e.g., Happy, Party), and view song data categorized by energy and danceability levels.
# The app uses ML (Linear Regression) to predict danceability and compares actual vs. predicted values.

import io
import os
import sqlite3 as sl
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for, send_file
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

db = "spotifyMusic.db"


def dance_energy_category(value):
    """
    Function to categorize the values(0.0-1.0) of energy and danceability to Low, Medium, or High.

    Parameters:
		value (float): The value to categorize.
	Returns:
		str: The category of the value.
	"""
    if value < 0.4:
        return "Low"
    elif value < 0.7:
        return "Medium"
    else:
        return "High"


def map_mood(row):
    """
    Map specific song features(valence, energy, tempo, danceability) to a specific mood.

    Parameters:
		value (dict): A dict containing the song features.
	Returns:
		str: The mood of the song.

    """
    if row["valence"] > 0.7 and row["energy"] > 0.7:
        return "Happy"
    elif row["tempo"] < 90 and row["energy"] < 0.5:
        return "Relaxing"
    elif row["tempo"] > 120 and row["energy"] > 0.7:
        return "Workout"
    elif row["danceability"] > 0.7 and row["energy"] > 0.7:
        return "Party"
    else:
        return "Other"


@app.route("/")
def home():
    """
    Render the home page.
    Fetches all unique genres from the database for user selection.
    """
    return render_template(
        "home.html", genres=db_get_genres(), message="Please select a genre."
    )


@app.route("/submit_genre", methods=["POST"])
def submit_genre():
    """
    Handle the genre selection from the user.
    Stores the selected genre in the session and redirects to the genre page.
    """
    session["genre"] = request.form["genre"]
    # Redirect to home if no genre is selected
    if not session["genre"]:
        return redirect(url_for("home"))
    return redirect(url_for("genre_data", genre=session["genre"]))


@app.route("/api/songs/<genre>")
def genre_data(genre):
    """
    Render the genre page with options for mood selection and visualization.
    """
    return render_template("genre.html", genre=genre)


@app.route("/submit_mood", methods=["POST"])
def submit_mood():
    """
    Handle the mood selection for the selected genre.
    Filters songs by the selected mood and displays song data(song name, artist, danceability and energy).
    """
    # Get mood from user input
    session["mood"] = request.form["mood"]

    if "genre" not in session:
        return redirect(url_for("home"))

    # Filter songs by genre
    # Load songs from the database for the selected genre
    df = db_create_dataframe(session["genre"])
    # iterate each row and map the mood for each song
    df["mood"] = df.apply(map_mood, axis=1)
    # Categorize energy and danceability for each song
    df["danceability"] = df["danceability"].apply(dance_energy_category)
    df["energy"] = df["energy"].apply(dance_energy_category)

    # Filter songs based on the selected mood
    filtered_songs = df[df["mood"] == session["mood"]][["track_name", "artists", "danceability", "energy"]]

    return render_template(
        "mood.html",
        genre=session["genre"],
        mood=session["mood"],
        songs=filtered_songs.to_dict(orient="records")  # Convert DataFrame to list of dictionaries for easy rendering
    )


@app.route("/fig/genre-popularity")
def fig_genre_popularity():
    """
    Generate a bar chart showing average popularity by genre.
    """
    fig = create_figure_genre_popularity()
    img = io.BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")


def create_figure_genre_popularity():
    """
    Create a plot showing the top 10 genres by average popularity.

    Returns:
	    Fig: The generated plot.
    """
    # Load all song data from the database
    df = db_create_dataframe()

    # Group data by genre and calculate the mean popularity for each genre
    genre_popularity = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False)

    # Select the top 10 genres
    top_genres = genre_popularity.head(10)

    # Create a bar chart
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(top_genres.index, top_genres.values, color="skyblue")

    # Add titles and labels
    ax.set(title="Average Popularity vs Top 10 Genres", xlabel="Genre", ylabel="Popularity")
    # Rotate x-axis labels for better visibility
    ax.set_xticklabels(top_genres.index, rotation=45, ha="right")

    fig.tight_layout()
    return fig


def create_figure_danceability_energy(genre):
    """
    Create a scatter plot of predicted danceability vs energy for the selected genre,
    and actual danceability vs energy.

    Parameters:
		genre (str): The selected genre.
	Returns:
		Fig: The generated plot.
    """
    df = db_create_dataframe(genre)
    # Train regression model to predict danceability based on energy, acousticness, and liveness
    X = df[["energy", "acousticness", "liveness"]]
    y = df["danceability"]

    model = LinearRegression()
    model.fit(X, y)

    # Predict danceability
    df["predicted_danceability"] = model.predict(X)

    # Categorize energy for each song
    df["energy_category"] = df["energy"].apply(dance_energy_category)
    # Colors for categories
    colors = {"Low": "red", "Medium": "orange", "High": "green"}

    # Create scatter plots
    fig = Figure(figsize=(10, 6))

    # Actual Danceability vs Energy
    ax1 = fig.add_subplot(2, 1, 1)
    # Iterate through each energy category and its associated color
    for category, color in colors.items():
        # Filter the DataFrame to include only rows where the energy_category matches the current category
        subset = df[df["energy_category"] == category]
        ax1.scatter(
            subset["energy"],
            subset["danceability"],
            label=category,
            color=color,
            alpha=0.7
        )
    ax1.set(title=f"Actual Danceability vs Energy ({genre})", xlabel="Energy", ylabel="Danceability")
    ax1.legend(title="Energy Level")
    ax1.grid(True)

    # Predicted Danceability vs Energy
    ax2 = fig.add_subplot(2, 1, 2)
    for category, color in colors.items():
        subset = df[df["energy_category"] == category]
        ax2.scatter(
            subset["energy"],
            subset["predicted_danceability"],
            label=category,
            color=color,
            alpha=0.7
        )
    ax2.set(title=f"Predicted Danceability vs Energy ({genre})", xlabel="Energy", ylabel="Predicted Danceability")
    ax2.legend(title="Energy Level")
    ax2.grid(True)

    fig.tight_layout()
    return fig


@app.route("/fig/danceability-prediction/<genre>")
def fig_danceability_prediction(genre):
    """
    Generate a scatter plot of predicted danceability vs energy for the selected genre,
    and actual danceability vs energy.
    """
    fig = create_figure_danceability_energy(genre)
    # Save the plot to BytesIO
    img = io.BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")


def db_create_dataframe(genre=None):
    """
    Load data from the database into a Pandas dataframe.
    """
    # Connect to the database
    conn = sl.connect(db)
    cursor = conn.cursor()

    stmt = "SELECT * FROM songs"
    params = None
    # Filter by genre if the genre is specified(i.e !=None).
    if genre:
        stmt += " WHERE track_genre = ?"
        params = (genre,)

    # Execute the stmt and fetch data(rows)
    cursor.execute(stmt, params if params else ())
    data = cursor.fetchall()
    # Fetch column names
    column_names = [description[0] for description in cursor.description]

    conn.close()

    # Convert results to a Pandas DataFrame
    df = pd.DataFrame(data, columns=column_names)
    return df


def db_get_genres():
    """
    Retrieve the genres from the database.
    """
    # Connect to the database
    conn = sl.connect(db)
    cursor = conn.cursor()

    # Execute stmt to get distinct genres
    stmt = "SELECT DISTINCT track_genre FROM songs"
    cursor.execute(stmt)

    # Fetch the genres
    genres = [row[0] for row in cursor.fetchall()]

    conn.close()

    return genres


@app.route('/<path:path>')
def catch_all(path):
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
