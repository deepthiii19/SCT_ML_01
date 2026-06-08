🏠 House Price Prediction — Linear Regression
SkillCraft Technology | Machine Learning Internship | Task 01

📌 Objective
Implement a Linear Regression model to predict house sale prices based on:

Square footage (living area + basement)
Number of bedrooms
Number of bathrooms (full + half)


📂 Dataset
Source: House Prices - Advanced Regression Techniques (Kaggle)
FileDescriptiontrain.csv1460 houses with features + sale pricetest.csv1459 houses for predictionsample_submission.csvSubmission formatdata_description.txtDescription of all 79 features

🔧 Features Used
FeatureDescriptionGrLivAreaAbove-grade living area (sq ft)TotalBsmtSFTotal basement area (sq ft)BedroomAbvGrNumber of bedrooms above gradeFullBathFull bathrooms above gradeHalfBathHalf bathrooms above grade

🛠️ Tech Stack

Python 3.x
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn


🚀 How to Run
1. Clone the repository
bashgit clone https://github.com/YOUR_USERNAME/SCT_ML_01.git
cd SCT_ML_01
2. Install dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn
3. Run the script
bashpython house_price_prediction.py

📊 Model Performance
MetricValueR² Score0.725MAE$30,693RMSE$45,929

The model explains 72.5% of the variance in house prices.


📈 Output Files Generated
FileDescriptioneda_plots.pngScatter plots of each feature vs Sale Price + Correlation heatmapmodel_evaluation.pngActual vs Predicted plot + Residual plotsubmission.csvFinal predicted prices for test set

🔍 Project Structure
SCT_ML_01/
│
├── house_price_prediction.py   # Main ML script
├── train.csv                   # Training data
├── test.csv                    # Test data
├── sample_submission.csv       # Submission format
├── data_description.txt        # Feature descriptions
├── eda_plots.png               # EDA visualizations
├── model_evaluation.png        # Model evaluation plots
├── submission.csv              # Predicted output
└── README.md                   # Project documentation

👨‍💻 Author
Internship: SkillCraft Technology — Machine Learning Track
Task: 01 of 04
