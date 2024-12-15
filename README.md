# **WNBA Shot Analysis Dashboard**

This project is an interactive dashboard for analyzing WNBA teams' shooting performance. Built using **Streamlit**, **Altair**, and **Pandas**, the dashboard provides insights into shot distributions, team comparisons, player performance, and in-game scoring evolution.

---

## **Features**

- **Key Metrics**:  
   - Total shots, success rates (overall and 3-point), and average shot distance.
- **Shots Distribution**:  
   - A heatmap and scatter plot to explore where shots are taken on the court.
- **Shot Evolution**:  
   - Analyze shooting success rates and total shots over time, dynamically for quarters or the full match.
- **Team Comparison**:  
   - Compare cumulative **point differences** or **accumulated points** between two teams for a selected game, including league average trends.
- **Player Performance**:  
   - Top player statistics presented in a bar chart with shooting percentages.

---

## **Data**

- **Dataset**:  
   The dataset `wnba-shots-2021.csv` contains WNBA shot-level data, including:
   - Player details  
   - Shot type and outcome  
   - Shot coordinates  
   - Quarter, time remaining, and team information  

---

## **Setup Instructions**

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/wnba-shot-dashboard.git
   cd wnba-shot-dashboard
