
### End-to-End Machine Learning Project Workflow

This notes outlines the process of executing an end-to-end machine learning (ML) project based on Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron, enriched with insights from modern ML practices.

**1. Look at the Big Picture**
- Frame the Problem: Define the ML task (e.g., supervised regression) in alignment with business goals. Clarify how the model will be used and specify the nature of the data.
- Define Success Criteria: Decide on the primary objectives (e.g., accuracy, interpretability, or efficiency) to guide model selection and tuning.
- Create a High-Level Plan: Develop a roadmap covering data collection, preprocessing, training, evaluation, and deployment to align stakeholders and maintain focus.

**2. Collect the Data**
Data is the foundation of ML projects, and its quality directly impacts results.

- Type- s of Data: Structured (databases), semi-structured (JSON), or unstructured (text, images), each requiring tailored handling techniques.
- Data Sources: Utilize public datasets, APIs, or internal company data.
- Data-Centric AI: Prioritize data quality through cleaning, augmentation, and enrichment for better model performance, as emphasized in recent research.

**3. Explore the Data**
Exploratory Data Analysis (EDA) helps uncover patterns, anomalies, and necessary transformations.

- Visualize the Data: Use tools like matplotlib and seaborn to reveal trends and relationships (e.g., correlations in housing prices).
- Handle Missing Values and Outliers: Detect gaps and anomalies using heatmaps and summary statistics, then address them through imputation or careful removal.

## Data Cleaning
In the book 3 options are listed:

1. housing.dropna(subset=["total_bedrooms"])    # option 1
2. housing.drop("total_bedrooms", axis=1)       # option 2
3. median = housing["total_bedrooms"].median()  
    housing["total_bedrooms"].fillna(median, inplace=True) # option 3

#### Code Blocks (Indented style)

I apply `median` to missing values in column total_bedrooms:

    median = housing["total_bedrooms"].median()  # option 3
    housing["total_bedrooms"].fillna(median, inplace=True)
    

#### While discovering data, I found ocean_proximity column has 5 rows for value `ISLAND`

    housing["ocean_proximity"].value_counts()
		
#### Results:
	<1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64

#### I remove those rows to overcome Bias issue:
    housing.drop(housing[housing['ocean_proximity'] == 'ISLAND'].index, inplace=True)

#### Now apply housing.info() , from the results you will notice:
	<class 'pandas.core.frame.DataFrame'>
    Int64Index: 20635 entries, 0 to 20639
	# There is index missing.
	# This issue will generate error in pandas and sklearn libraries. 

#### To solving this issue reset dataframe indexing:
    housing.dropna(inplace=True) 
    housing.reset_index(drop=True, inplace=True)

#### Results: enhace regression model prediction error, as:
		 Befor clear ['ocean_proximity'] == 'ISLAND': 
		 		root mean square error: 84056.18763327331
		 After clear ['ocean_proximity'] == 'ISLAND': 
		 		root mean square error: 83786.95955423421

	Befor modification: 
			Linear Regression Prediction error: 69957.72079714121
	After modification: 
			Linear Regression Prediction error: 69725.06911394256

	Befor: Random forest prediction error 18694.75574646658
	After: Random forest prediction error 18577.559149145716

    

#### Notice: Unfortunatly Random Forest Regressor prediction error goes high:
		Befor: Test set prediction error: 47927.65438939967
		After: Test set prediction error: 49420.82944683894


**Conclusion**
A structured workflow, from problem framing to deployment, ensures ML projects deliver practical value. Géron’s guidelines, combined with modern insights, help create robust and impactful models.
I had indecator to select  classification model to solve this problem.


TODO:
=============
- [ ] Make model base on columns (total_rooms, total_bedrooms, population, households) as they has good correlation.
- [ ] Make model base on classify geolocation data. 
