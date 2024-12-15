from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data
similarity_matrix = pd.read_csv('filtered_similarity_matrix.csv', index_col=0)
rating_matrix = pd.read_csv('rating_matrix.csv', index_col=0)
popularity_ranking = pd.read_csv('popularity_ranking.csv', index_col=0)
movie_metadata = pd.read_csv('movie_metadata.csv')

# Select the top 100 most rated movies globally
most_rated_movies = rating_matrix.notna().sum().sort_values(ascending=False).head(100).index
sample_movies = list(most_rated_movies)


def myIBCF(newuser):
    """
    Predicts ratings for movies that the new user has not rated using the IBCF approach.
    Then, recommends the top 10 movies based on predictions.
    """

    import pandas as pd
    import numpy as np

    # Validate that the input contains only NA or integers 1 to 5
    if not newuser.isna().all():  # Skip if all values are NA
        invalid_values = newuser.dropna()[~newuser.dropna().isin([1, 2, 3, 4, 5])]
        if not invalid_values.empty:
            raise ValueError(
                f"Invalid ratings detected: {invalid_values.to_dict()}. "
                "Ratings must be NA or integers 1, 2, 3, 4, or 5."
            )

    # Load similarity matrix, rating matrix, and popularity ranking internally
    similarity_matrix = pd.read_csv('filtered_similarity_matrix.csv', index_col=0)
    rating_matrix = pd.read_csv('rating_matrix.csv', index_col=0)
    popularity_ranking = pd.read_csv('popularity_ranking.csv', index_col=0)

    # Select the top 100 movies with the most ratings
    most_rated_movies = rating_matrix.notna().sum().sort_values(ascending=False).head(100).index
    similarity_matrix = similarity_matrix.loc[most_rated_movies, most_rated_movies]
    rating_matrix = rating_matrix[most_rated_movies]

    # Ensure newuser is a vector aligned with the selected movies
    all_movies = rating_matrix.columns
    if len(newuser) != len(all_movies):
        # Create a vector with NA for missing movies
        newuser_full = pd.Series(data=np.nan, index=all_movies)
        newuser_full.update(newuser)  # Update with provided ratings
        newuser = newuser_full

    # Initialize predictions for all movies as NA
    predictions = pd.Series(index=newuser.index, dtype='float64')

    # Compute predicted ratings for each movie
    for movie_i in newuser.index:
        if not pd.isna(newuser[movie_i]):  # Skip movies already rated by the user
            continue

        # Get similarity values for movie_i
        movie_similarities = similarity_matrix.loc[movie_i]

        # Get movies rated by the user
        rated_movies = newuser[newuser.notna()].index
        rated_similarities = movie_similarities[rated_movies]

        # Compute numerator and denominator for the prediction formula
        numerator = np.sum(rated_similarities * newuser[rated_movies])
        denominator = np.sum(rated_similarities)

        # Compute the predicted rating for movie_i
        predictions[movie_i] = numerator / denominator if denominator != 0 else np.nan

    # Filter predictions for the top 10 recommendations
    top_predictions = predictions.dropna().sort_values(ascending=False).head(10)

    # If fewer than 10 movies have predictions, use the popularity-based ranking
    if len(top_predictions) < 10:
        already_rated_movies = newuser[newuser.notna()].index
        remaining_movies = popularity_ranking.loc[
            ~popularity_ranking.index.isin(already_rated_movies)
        ].head(10 - len(top_predictions))
        additional_predictions = pd.Series(
            remaining_movies['score'], index=remaining_movies.index
        )
        top_predictions = pd.concat([top_predictions, additional_predictions])

    # Return the top 10 recommendations as a DataFrame
    top_10_recommendations = pd.DataFrame({
        'MovieID': top_predictions.index,
        'PredictedRating': top_predictions.values.round(4)  # Round to 4 decimal places
    }).reset_index(drop=True)

    return top_10_recommendations

# Update sample_movies to reflect the top 100 most rated movies globally
most_rated_movies = rating_matrix.notna().sum().sort_values(ascending=False).head(100).index
sample_movies = list(most_rated_movies)



@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home route: Displays the movie rating form with recommendations (if available).
    """
    recommendations = []
    error_message = None

    if request.method == 'POST':
        try:
            user_ratings = {
                movie: float(request.form.get(movie)) if request.form.get(movie) else None
                for movie in sample_movies
            }

            invalid_ratings = {
                movie: rating for movie, rating in user_ratings.items()
                if rating is not None and (rating < 1 or rating > 5 or not rating.is_integer())
            }
            if invalid_ratings:
                error_message = f"Invalid ratings detected: {invalid_ratings}. Ratings must be integers between 1 and 5."
            else:
                user_series = pd.Series(user_ratings)
                recommendations = myIBCF(user_series).to_dict(orient='records')

        except ValueError as e:
            error_message = str(e)
        except Exception as e:
            error_message = "An unexpected error occurred."

    movies_with_metadata = movie_metadata[movie_metadata['MovieID'].isin(sample_movies)].to_dict(orient='records')
    return render_template('index.html', movies=movies_with_metadata, recommendations=recommendations, error=error_message)


if __name__ == '__main__':
    app.run(debug=True)
