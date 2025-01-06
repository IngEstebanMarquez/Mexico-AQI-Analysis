# Mexico-AQI-Analysis
This project aims to predict and analyze air quality levels across various locations in Mexico using environmental data such as NOx, O3, CO, and other key pollutants. The model leverages the power of machine learning to forecast AQI values and detect anomalies in air quality, helping to identify regions with the best and worst air quality.

## Overview
This repository contains the code for predicting AQI levels using machine learning techniques, analyzing air quality data, detecting anomalies, and ranking locations based on their pollution levels. The dataset includes environmental data from monitoring stations across Mexico, and the project uses Random Forest Regressor, Isolation Forest, and other methods to make predictions and detect unusual changes in air quality.

## Data Sources
- **URL**: [Kaggle - Mexico Air Quality Dataset](https://www.kaggle.com/datasets/elianaj/mexico-air-quality-dataset)
- **Author**: Eliana Kai Juarez
- **Description**: Hourly air pollution and weather data for 100+ stations across Mexico
- **License**: CC0: Public Domain

## Analysis Highlights
- **AQI Prediction**: Using Random Forest Regressor to forecast AQI based on environmental factors.
- **Anomaly Detection**: Identifying anomalies in air quality with Isolation Forest.
- **Air Quality Ranking**: Ranking cities and locations by their average AQI, identifying the cleanest and most polluted areas.
- **Visualizations**: Informative visualizations that include scatter plots, bar charts, and maps to show AQI trends and geographical patterns.

## Key Features
- **Predict AQI** using environmental variables like NOx, O3, CO, etc.
- **Detect anomalies** in air quality with Isolation Forest.
- **Rank locations** based on their average AQI levels.
- **Visualize data** through graphs, maps, and other interactive charts.
- **Air Quality Categorization** to easily identify regions with good or hazardous air quality.

## How to Explore
To explore this project, you can check out the following files in this repository:

1. **Mexico Air Quality Analysis.ipynb**: A Jupyter notebook containing the full code and analysis.
2. **Mexico Air Quality Analysis.py**: The Python script. Code-only version of the notebook.
3. **stations_daily.csv**: One of the original datasets used for analysis.
4. **stations_rsinaica**: One of the original datasets used for analysis.

## Dataset 
The dataset is stored in a .rar file. Please extract it using a tool like [WinRAR](https://www.win-rar.com/) or [7-Zip](https://www.7-zip.org/) to access the original CSV file.

## Tools and Technologies
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Folium)
- Jupyter Notebooks for data exploration and visualization
- Machine Learning: Random Forest Regressor, Isolation Forest
- Data Visualization: Matplotlib, Seaborn, Folium

## Notes
- This project focuses on predicting AQI and detecting anomalies in air quality. It uses environmental factors like NOx, O3, CO, and other pollutants.
- Make sure you have the necessary libraries installed (`pandas`, `matplotlib`, `seaborn`, `scikit-learn`) to run the code successfully.
- Make sure to extract the dataset properly before running the analysis.
- For more details on the methodology or to contribute, check the issues or feel free to reach out!

---
Made by **Esteban Márquez**.  
Connect with me on [LinkedIn](https://www.linkedin.com/in/ing-esteban-marquez)  
Feel free to connect or ask questions!
