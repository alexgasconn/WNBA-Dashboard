import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="WNBA Shot Analysis Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded")

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_dir = os.path.join(base_dir, "logos")
csv_dir = os.path.join(base_dir, "wnba-shots-2021.csv")


# Load data
@st.cache_data
def load_data():
    return pd.read_csv(csv_dir)

data = load_data()


##############################################
# Prepare Data
basket_x = 25
basket_y = 0

# Filter and add columns
data = data[~data['shooting_team'].isin(['Team Wilson', 'Team Stewart'])]
data = data[~data['qtr'].isin([5, 6])]

data['distance'] = np.sqrt((data['coordinate_x'] - basket_x)**2 + (data['coordinate_y'] - basket_y)**2)
data['time_category'] = data['quarter_seconds_remaining'].apply(lambda x: 'Clutch' if x <= 30 else 'Normal')

data = data[data['coordinate_x'] >= 0].copy()
data = data[data['coordinate_y'] >= 0].copy()


def categorize_shot(shot_type):
    if 'Jump Shot' in shot_type:
        return 'Jump Shots'
    elif 'Layup' in shot_type:
        return 'Layups'
    elif 'Hook' in shot_type:
        return 'Hook Shots'
    else:
        return 'Finishing Shots'
data['shot_category'] = data['shot_type'].apply(categorize_shot)

data['winning_status'] = data.apply(lambda row: 'Winning' if row['home_score'] > row['away_score']else 'Losing', axis=1)

data['winning_team_name'] = data.apply(lambda row: row['home_team_name'] if row['home_score'] > row['away_score'] 
                                       else (row['away_team_name'] if row['away_score'] > row['home_score'] else 'Tie'), axis=1)

data["winning_team"] = data.apply(lambda row: "Home" if row["home_score"] > row["away_score"] else "Away", axis=1)

data['point_difference'] = data['home_score'] - data['away_score']

def get_player_name(df):
    full_names = []
    for desc in df['desc'].str.split(" "):
        name = desc[0]
        surname = desc[1]
        surname2 = ""
        if desc[2] not in ["misses", "makes", "blocks"]:
            surname2 = desc[2]

        full_name = f"{name} {surname} {surname2}".strip()
        full_names.append(full_name)
    df['player'] = full_names
    return df
data = get_player_name(data)

def get_team_logo(team_name):
    logo_path = os.path.join(logo_dir, f"{team_name}.png")
    if os.path.exists(logo_path):
        return logo_path
    return None


##############################################
# Sidebar: Filters
with st.sidebar:
    st.title("🏀 WNBA Shot Dashboard")
    st.markdown("Analyze WNBA teams' performance using interactive visualizations.")

    # Team filter
    selected_team = st.selectbox("Select a Team", options=["All"] + list(data['shooting_team'].unique()))

    # Quarter filter
    selected_quarter = st.selectbox("Select a Quarter", options=["All"] + [str(q) for q in sorted(data['qtr'].dropna().unique())])

    # Shot outcome filter
    selected_outcome = st.selectbox(
        "Shot Outcome", options=["All", "Made", "Missed"],
        format_func=lambda x: "All" if x == "All" else "Made" if x == "Made" else "Missed"
    )

    # Game location filter
    selected_game_location = st.radio(
        "Select Game Location",
        options=["All", "Home Games", "Away Games"],
        index=0  # Default is "All"
    )

    # Shot type filter
    selected_shot = st.selectbox("Select a shot type", options=["All"] + list(data['shot_category'].unique()))

##############################################
# Filters
filtered_data = data.copy()

# Apply team filter
if selected_team != "All":
    filtered_data = filtered_data[filtered_data['shooting_team'] == selected_team]

# Apply quarter filter
if selected_quarter != "All":
    filtered_data = filtered_data[filtered_data['qtr'] == int(selected_quarter)]

# Apply shot outcome filter
if selected_outcome == "Made":
    filtered_data = filtered_data[filtered_data['made_shot'] == True]
elif selected_outcome == "Missed":
    filtered_data = filtered_data[filtered_data['made_shot'] == False]

# Apply game location filter
if selected_game_location == "Home Games":
    if selected_team != "All":
        filtered_data = filtered_data[filtered_data['home_team_name'] == selected_team]
    else:
        filtered_data = filtered_data[filtered_data['home_team_name'].notna()]
elif selected_game_location == "Away Games":
    if selected_team != "All":
        filtered_data = filtered_data[filtered_data['away_team_name'] == selected_team]
    else:
        filtered_data = filtered_data[filtered_data['away_team_name'].notna()]

# Apply shot category filter
if selected_shot != "All":
    filtered_data = filtered_data[filtered_data['shot_category'] == selected_shot]









########################################### [LOGOS] #####################################################################
col1, col2 = st.columns([1, 9], gap="medium")

with col1:
    if selected_team != "All":
        team_logo_path = get_team_logo(selected_team)
        if team_logo_path:
            st.image(team_logo_path, use_container_width=True, caption=selected_team)
    else:
        st.image(os.path.join(logo_dir, "WNBA.png"), use_container_width=True)


with col2:
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>WNBA Shot Analysis</h1>
        </div>
        """, unsafe_allow_html=True)

            
########################################### [GRAPHICS] #####################################################################

col = st.columns((2, 4.5, 4.5), gap="medium")

# Section 1: Metrics
with col[0]:
    st.markdown("#### Key Metrics")
    
    quarter_data = filtered_data.copy()
    if selected_quarter != "All":
        quarter_data = quarter_data[quarter_data['qtr'] == int(selected_quarter)]
    
    total_shots = quarter_data.shape[0]
    successful_shots = quarter_data[quarter_data['made_shot'] == True].shape[0]
    success_percentage_total = round((successful_shots / total_shots * 100), 2) if total_shots > 0 else 0

    three_point_data = quarter_data[quarter_data['distance'] > 23.75]  # Assume 23.75 feet as 3-point line
    total_threes = three_point_data.shape[0]
    successful_threes = three_point_data[three_point_data['made_shot'] == True].shape[0]
    success_percentage_threes = round((successful_threes / total_threes * 100), 2) if total_threes > 0 else 0

    avg_distance = round(quarter_data['distance'].mean(), 2) if total_shots > 0 else 0

    st.metric("Total Shots", total_shots)
    st.metric("Success Rate (Total) (%)", success_percentage_total)
    st.metric("Total 3-Point Shots", total_threes)
    st.metric("Success Rate (3-Point) (%)", success_percentage_threes)
    st.metric("Average Shot Distance (ft)", avg_distance)




# Function to create basketball court edges as Altair lines
def court_edges():
    edges_data = pd.DataFrame({
        'x': [0, 50, 50, 0, 0],  # Define the outer edges of the court
        'y': [0, 0, 85, 85, 0],  # Top and bottom edges, and left and right edges
        'group': [1, 1, 1, 1, 1]  # Group to ensure a continuous line
    })
    edges_outline = alt.Chart(edges_data).mark_line(color='black', opacity=0.7).encode(
        x='x:Q',
        y='y:Q',
        detail='group:N'
    )
    return edges_outline


########################################### [Shot distribution graphic] #####################################################################

# Section 2: Heatmap
with col[1]:
    st.markdown("#### Shots Distribution")

    heatmap = alt.Chart(filtered_data).mark_rect().encode(
        x=alt.X('coordinate_x:Q', bin=alt.Bin(maxbins=25), title=None, 
            scale=alt.Scale(domain=[0, 50]),
            axis=alt.Axis(labels=False, ticks=False)
        ),
        y=alt.Y('coordinate_y:Q', bin=alt.Bin(maxbins=25), title=None,
            scale=alt.Scale(domain=[0, 85]),
            axis=alt.Axis(labels=False, ticks=False)
        ),
        color=alt.Color('count():Q', scale=alt.Scale(scheme='greens'), title='Shot Frequency'),
        tooltip=[
            alt.Tooltip('count()', title='Shot Count'),
            alt.Tooltip('average(distance):Q', title='Avg Distance'),
            alt.Tooltip('average(made_shot):Q', title='Shot Success Rate (%)', format='.1%')
        ]
    ).properties(width=700, height=400)

    # Scatter Plot Toggle
    scatter_toggle = st.checkbox("Display shots position")
    scatter = None
    if scatter_toggle:
        scatter = alt.Chart(filtered_data).mark_circle(size=10, opacity=0.6).encode(
            x=alt.X('coordinate_x:Q', title=None,
                scale=alt.Scale(domain=[0, 50]),
                axis=alt.Axis(labels=False, ticks=False)
            ),
            y=alt.Y('coordinate_y:Q', title=None,
                scale=alt.Scale(domain=[0, 85]),
                axis=alt.Axis(labels=False, ticks=False)
            ),
            color=alt.value('darkgreen'),
            tooltip=['player', 'shot_category', 'distance', 'made_shot']
        )

    # Combine Heatmap, Scatter (if toggled), and Court Edges
    court_outline = court_edges()
    if scatter:
        combined_chart = heatmap + scatter + court_outline
    else:
        combined_chart = heatmap + court_outline

    st.altair_chart(combined_chart, use_container_width=True)




########################################### [Shot Analysis Over Quarter Time] #####################################################################

# Section 3: Shot Analysis Over Quarter Time
with col[2]:
    st.markdown("#### Shot evolution during time")

    time_granularity = st.radio(
        "Select Time Granularity:",
        options=["By Quarter", "Full Match"],
        index=0, 
        horizontal=True
    )

    if selected_quarter != "All":
        quarter_data = filtered_data[filtered_data['qtr'] == int(selected_quarter)]
    else:
        quarter_data = filtered_data

    if time_granularity == "By Quarter":
        quarter_data['time_bin'] = (quarter_data['quarter_seconds_remaining'] // 60) * 60
    else:
        quarter_data['time_bin'] = (quarter_data['game_seconds_remaining'] // 60) * 60 

    shot_analysis = quarter_data.groupby(['time_bin']).agg(
        total_shots=('made_shot', 'size'),
        made_shots=('made_shot', 'sum')
    ).reset_index()

    shot_analysis['made_shot_percentage'] = round((shot_analysis['made_shots'] / shot_analysis['total_shots']) * 100, 2)

    line_chart = alt.Chart(shot_analysis).mark_line(point=True).encode(
        x=alt.X('time_bin:O', title="Time Remaining in Quarter (Seconds)", sort="descending", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('made_shot_percentage:Q', title="Shooting Success Percentage (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.value('darkgreen'),
        tooltip=['time_bin', 'made_shot_percentage']
    ).properties(width=600, height=300)

    bar_chart = alt.Chart(shot_analysis).mark_bar(opacity=0.5).encode(
        x=alt.X('time_bin:O', title="Time Remaining in Quarter (Seconds)", sort="descending", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('total_shots:Q', title="Total Number of Shots", axis=alt.Axis(grid=False)),
        color=alt.value('green'),
        tooltip=['time_bin', 'total_shots']
    ).properties(width=600, height=300)

    combined_chart = alt.layer(bar_chart, line_chart).resolve_scale(
        y='independent'
    ).properties(
        width=700,
        height=400)

    st.altair_chart(combined_chart, use_container_width=True)






########################################### [POINTS DIFFERENCE] #####################################################################
if selected_team != "All":
    team_games = data[
        (data['home_team_name'] == selected_team) | (data['away_team_name'] == selected_team)
    ].drop_duplicates(subset=['game_id'])
else:
    team_games = data.drop_duplicates(subset=['game_id'])

# Create game mapping for dropdown options
game_mapping = {
    game_id: f"{row['home_team_name']} vs {row['away_team_name']}"
    for game_id, row in team_games.set_index('game_id').iterrows()
}
reverse_game_mapping = {v: k for k, v in game_mapping.items()}


# Calculate league average for accumulated points with a moving average
league_avg_points = data.groupby('game_seconds_remaining').apply(
    lambda df: (df['home_score'].mean() + df['away_score'].mean()) / 2
).reset_index(name='league_avg_score')

league_avg_points['league_avg_score'] = league_avg_points['league_avg_score'].rolling(window=5, min_periods=1).mean()


if game_mapping:
    selected_game_name = st.selectbox("Select Game", options=list(game_mapping.values()), key="game_selection")
    selected_game_id = reverse_game_mapping[selected_game_name]

    # Fetch selected game details
    selected_game = data[data['game_id'] == selected_game_id].iloc[0]
    game_df = data[data['game_id'] == selected_game_id]
    home_team = selected_game['home_team_name']
    away_team = selected_game['away_team_name']

    # Display team logos
    col2, col3 = st.columns([7, 3], gap="medium")

    # Slider for time left
    time_min = int(game_df['game_seconds_remaining'].min())
    time_max = int(game_df['game_seconds_remaining'].max())
    time_range = st.slider("Select Time Range (Seconds Remaining):",
    min_value=time_max, max_value=time_min,
    value=(time_max, time_min), step=10)

    # Filter data based on selected time range
    filtered_game_df = game_df[
    (game_df['game_seconds_remaining'] >= time_range[0]) &
    (game_df['game_seconds_remaining'] <= time_range[1])]

    # Toggle for Point Difference or Accumulated Points
    view_option = st.radio(
        "Select Chart Type",
        options=["Point Difference", "Accumulated Points"],
        index=0,  # Default selection is "Point Difference"
        horizontal=True)




#################### [Point Difference] ################################
if view_option == "Point Difference":
    filtered_game_df['winning_status'] = filtered_game_df.apply(
        lambda row: home_team if row['home_score'] > row['away_score'] else (
            away_team if row['away_score'] > row['home_score'] else 'Tie'
        ),axis=1)

    chart = alt.Chart(filtered_game_df).mark_area(opacity=0.9).encode(
        alt.X('game_seconds_remaining:Q',
            title='Game Time Remaining (Seconds)',
            scale=alt.Scale(reverse=True)  # Invert the X-axis
        ),
        alt.Y('point_difference:Q', title='Cumulative Point Difference'),
        alt.Color('winning_status:N', title="Team Leading",
            scale=alt.Scale(
                domain=[home_team, away_team],
                range=['lightgreen', 'darkgreen'])
        ),
        tooltip=['shooting_team:N', 'point_difference:Q', 'qtr:N'])



#################### [Accumulated Points Chart] ################################
elif view_option == "Accumulated Points":
    team_chart = alt.Chart(filtered_game_df).transform_fold(
        ['home_score', 'away_score'], as_=['OriginalTeam', 'Score']
    ).transform_calculate(
        # Replace "home_score" and "away_score" with teh team names
        Team="datum.OriginalTeam === 'home_score' ? '" + home_team + "' : '" + away_team + "'"
    ).mark_line().encode(
        alt.X('game_seconds_remaining:Q', title='Game Time Remaining (Seconds)',
            scale=alt.Scale(reverse=True)),
        alt.Y('Score:Q', title='Accumulated Points'),
        alt.Color('Team:N',title="Team",
            scale=alt.Scale(
                domain=[home_team, away_team],
                range=['lightgreen', 'darkgreen'])
        ),tooltip=['Team:N', 'Score:Q'])

    league_avg_chart = alt.Chart(league_avg_points).mark_line(
        color='gray', strokeDash=[5, 5]
    ).encode(
        alt.X('game_seconds_remaining:Q'),
        alt.Y('league_avg_score:Q', title='Accumulated Points'),
        tooltip=['league_avg_score:Q'])
    chart = team_chart + league_avg_chart

with col2:
    st.altair_chart(chart.properties(width=600, height=400), use_container_width=True)



#################### [Player Performance Bar Chart] ################################
with col3:
    st.markdown("#### Player Performance")

    # Calculate player statistics
    player_stats = filtered_game_df.groupby(['player', 'shooting_team']).agg(
        points=('shot_value', 'sum'),
        total_shots=('made_shot', 'size'),
        made_shots=('made_shot', 'sum')
    ).reset_index()
    player_stats['shooting_percentage'] = ((player_stats['made_shots'] / player_stats['total_shots']) * 100).round(1)

    player_stats['points_and_percentage'] = (
        player_stats['points'].astype(str) + " pts (" +
        player_stats['shooting_percentage'].round(1).astype(str) + "%)")

    top_players = player_stats.sort_values(by='points', ascending=False).head(10)

    bar_chart = alt.Chart(top_players).mark_bar().encode(
        x=alt.X('points:Q', title='Points'),
        y=alt.Y('player:N', title='Player', sort='-x'),
        color=alt.Color(
            'shooting_team:N',
            title='Team',
            scale=alt.Scale(
                domain=[home_team, away_team],
                range=['lightgreen', 'darkgreen']
            ),legend=None
        ),
        tooltip=['player:N', 'shooting_team:N', 'points:Q', 'shooting_percentage:Q']
    ).properties(
        title="Top Players by Points",
        width=300,
        height=400)

    text_labels = alt.Chart(top_players).mark_text(dx=5, align='left').encode(
        x=alt.X('points:Q'),
        y=alt.Y('player:N', sort='-x'),
        text=alt.Text('points_and_percentage'),
        color=alt.value('black'))

    combined_chart = bar_chart + text_labels
    st.altair_chart(combined_chart, use_container_width=True)