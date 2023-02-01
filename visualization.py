from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Patients In A Hospital'),
    html.H3('Size of Bubble Indicates Cost of Servicing Patient'),
    dcc.RadioItems(
        id='animations-x-selection',
        value='GDP - Scatter',
    ),
    dcc.Loading(dcc.Graph(id="animations-x-graph"), type="cube")
])

# Load the data into a pandas dataframe
df2 = pd.read_csv('inpatient_discharges_first_six.csv')

# Split the data into training and testing sets
training_data = df2[:int(len(df2) * 0.8)]
testing_data = df2[int(len(df2) * 0.8):]

# Convert the DRG descriptions into numerical features using Tf-Idf
vectorizer = TfidfVectorizer()
training_features = vectorizer.fit_transform(training_data['DRG Description'])
testing_features = vectorizer.transform(testing_data['DRG Description'])

# Train a logistic regression model to predict the length of stay
model = LogisticRegression()
model.fit(training_features, training_data['LOS'])

# Make predictions on the testing data
predictions = model.predict(testing_features)

# Evaluate the model's accuracy
accuracy = model.score(testing_features, testing_data['LOS'])
print("Accuracy:", accuracy)


@app.callback(
    Output("animations-x-graph", "figure"),
    Input("animations-x-selection", "value"))
def display_animated_graph(selection):
    df = px.data.gapminder()

    df = df[df['continent'] != 'Oceania']
    df = df[df['continent'] != 'Americas']
    df = df[df['country'] != 'China']
    df = df[df['country'] != 'India']
    df = df[df['country'] != 'Japan']

    df['year'] = (df['year'] - 1952) // 5 + 1
    df.rename(columns={'year': 'Day'}, inplace=True)
    df.rename(columns={'lifeExp': 'Financial Benefit'}, inplace=True)
    df.rename(columns={'gdpPercap': 'Patient Experience'}, inplace=True)
    df.rename(columns={'continent': 'AI Optimization Level'}, inplace=True)
    df.rename(columns={'pop': 'Cost'}, inplace=True)
    df.rename(columns={'country': 'id'}, inplace=True)

    df['AI Optimization Level'].replace('Africa', 'Low', inplace=True)
    df['AI Optimization Level'].replace('Europe', 'High', inplace=True)
    df['AI Optimization Level'].replace('Asia', 'Medium', inplace=True)

    df.loc[df['AI Optimization Level'] == 'Low', 'Financial Benefit'] = 30 + (df['Day'] / 100) * 10
    # round the values in the 'Financial Benefit' column to 2 decimal places
    df['Financial Benefit'] = np.round(df['Financial Benefit'], 2)

    animations = {
        'GDP - Scatter': px.scatter(
            df, x="Patient Experience", y="Financial Benefit", animation_frame="Day",
            animation_group="id", size="Cost", color="AI Optimization Level",
            log_x=True, size_max=55,
            range_x=[100, 100000], range_y=[25, 90])
    }

    return animations[selection]


if __name__ == "__main__":
    app.run_server(debug=True)
