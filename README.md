![Alt](./Images/pexels-photo-27173251.webp)

### <span style="color: rgb(96, 157, 179);">**Predicting Purchase Likelihood for Better Conversion Rates at MarketGenix**</span>

##### **<span style= "color: rgb(171, 211, 226);"> Objective </span>**  
At MarketGenix, the goal for 2025 is to increase conversions by focusing on high-potential leads. As part of the data science team, I was tasked with building a predictive model that leverages Google Analytics data to forecast the likelihood of purchases. This will enable the marketing team to target prospects who are most likely to convert, optimizing efforts and driving business growth.

##### **<span style= "color: rgb(171, 211, 226);"> Key Features </span>**  
The model uses various data dimensions to assess purchase likelihood:  
- **Engagement Metrics**: `Engagement_Level`, `Likes`, `Shares`, `Comments`, `Clicks`, `Time_Spent_on_Platform`  
- **Behavioral Insights**: `Engagement_with_Ads`, `Purchase_History`  
- **Outcome**: `Purchase_Likelihood`

##### **<span style= "color: rgb(171, 211, 226);"> Task Summary </span>**  

This project predicts **Purchase Likelihood (binary: 0/1)** using various classification models. Key steps and outcomes:

- **Scaling**:  
  - **RobustScaler** was used due to data skewness and the presence of outliers, based on interquartile range (IQR) analysis.  
  - Logarithmic transformations were applied to reduce skewness in select features.  

- **Scoring Metric**:  
  - **Recall** was prioritized to minimize false negatives, aligning with the marketing goal of maximizing lead conversions.  

- **Best Model**:  
  - **RidgeClassifierCV with PCA**:  
    - **Recall**: 0.7206  
    - **F1-Score**: 0.5568  
    - **Accuracy**: 0.48  

This model excelled by leveraging PCA for dimensionality reduction and Ridge for robust regularization, ensuring the highest recall for true positive identification.