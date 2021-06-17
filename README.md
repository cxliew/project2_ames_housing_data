# Project 2: Ames Housing Data & Kaggle Challenge

### Problem Statement

Due to the increasing population in Ames Iowa, there is an increasing demand for housing for the residents in Ames Iowa. As it is the same in many place, the sales price of a house typically will vary according to its types and features. As a data scientist working in a private construction company in Ames Iowa, we are interested to first explore the association between residential features and its sales prices. This will allows us to gain an insight to optimize the construction plan to include those features that are correlated to prices, which will translate to higher sales profits.

---
### Background

Houses styles in United states varies across different parts of the states depending on the location you are in. In 1991, many of the houses in United States are landed property with 60% of the USA land is a private land,[[1]](https://www.summitpost.org/public-and-private-land-percentages-by-us-states/186111) thus making the construction a lucrative business. As the human population are increasing,this increases the demand of the need of new houses to be build and existing houses to be remodeled making the knowledge of consumer desirables features an important asset.

The need of new houses also extend to Ames, Iowa, which is a city in Story County,Iowa,[[2]](https://www.niche.com/places-to-live/ames-story-ia/) which is located in the midwest region of the United States. The housing unit with owner-occupied in 2019 is 40.7% with a median value of these housing units of USD205,900.[[3]](https://www.census.gov/quickfacts/amescityiowa) With the home appreaction in Ames increases quickly up to 28.1% in the last 10 years [[4]](https://www.bestplaces.net/housing/city/iowa/ames), Awes, Iowa faced a challenge of purchasing an affordable housing resulting in an increase in remodeling run down house leading to the disinterest of developers in Ames city[[5]](https://www.thegazette.com/regional-development/iowas-affordable-housing-struggle/). With the need of more housing, our private construction company in Ames, Iowa would like to gain further insight to explore the association between residential that are correlated to prices. This allows the company to integrate this domain knowledge of consumer desirables features to build more affordable housing, thus translating to higher sales profits. Hence, as a data scientist in this company, I will be handling on this project and our goals are the following:

1. To identify the features that are highly correlated to sale prices (from both domain knowledge and technical analysis)

2. To build a good model to estimate and predict the sales price of housing in Ames Iowa

By knowing what are the key features that affect the sales price, our construction company will be able to use this insight to make informed business decisions, such as to build desirable affordable houses that are highly marketable and advise potential house owners to renovate their house to include features that are likely to increase sales price prior selling.

---
### Data Dictionary

Data source: [Ames Housing Data](https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge/data) obtained from database [Ames Iowa Assessor's Office](http://www.cityofames.org/assessor/)

|Feature|Type|Dataset|Description|
|:------|:---:|:---:|:---|
|overall_qual|integer|Housing|The overall material and finish of the house| 
|exter_qual|integer|Housing|Evaluates the quality of the material on the exterior| 
|bsmt_qual|integer|Housing|Evaluates the height of the basement| 
|total_bsmt_sf|integer|Housing|Total square feet of basement area|
|heating_qc|integer|Housing|Heating quality and condition| 
|gr_liv_area|integer|Housing|Above grade (ground) living area square feet|
|full_bath|integer|Housing|Full bathrooms above grade (ground)|  
|kitchen_qual|integer|Housing|Kitchen quality| 
|fireplace_qu|integer|Housing|Fireplace quality| 
|garage_finish|integer|Housing|Interior finish of the garage| 
|garage_area|integer|Housing|Size of garage in square feet| 
|log_saleprice|float|Housing|Logarithm of the sale price of the house| 
|age|integer|Housing|Age of the house|
|bsmt_qual_total|integer|Housing|Combine effect of total basement area and its quality| 
|arage_area_finish|integer|Housing|Combine effect of garage area and garage finish|
|fire_heat_qc|integer|Housing|Combine effect of Fireplace quality and heating quality| 
|totrms_abvgrd|integer|Housing|Total rooms above ground|
|ms_zoning|integer|Housing|The general zoning classification of the sale|
||||- FV  Floating Village Residential|        
||||- I   Industrial|   
||||- RH  Residential High Density|        
||||- RL  Residential Low Density|      
||||- RM  Residential Medium Density| 

---
### Data Cleaning

**For both housing_data and test_housing_data:**
* Rename the column for coherent and easier navigation and drop the columns of 'id' and 'pid'.
* Rename the values of ms_zoning column to remove '(all)' and '(agr)'.
* Fill the missing values for column having missing values in the alley, basement, fence, fireplace, frontage, garage, masonry, pool and miscellaneous features columns based on the similar characteristics of the missing values.
* The missing values that are ordinal and nominal category has been converted to integer
* The numeric columns (float) has been converted to integer
* The ordinal columns have been convereted to integer

**For test_housing_data:**
* The additional column of missing values in elecrical has been filled.

---
## Exploratory Data Analysis

**The data consists of 8 main categories which are as followed:-**
1. Sales
2. Physical House Characteristics
3. House Material and Foundation
4. Rooms
5. Basement
6. Garage
7. Electrical and Heating
8. External Facilities

We will further explore each category in-depth with consideration of several of potential buyer interest as listed in the property search engine [[6]](https://www.movoto.com/ames-ia/new-7/)[[7]](https://desmoines.homefront.com/homesearch/ames-ia)[[8]](https://www.zillow.com/homes/)

The features of interest are as followed:-
1. Main Interest - Bedroom, Bath, Above Ground Living Area, Year Built, Lot Area, Location
2. Other Considerations - Ms zoning, Basement, Central Air, Heating, Fireplace, Garage Cars, Exterior Roofing, Exterior House (general), Fencing, Utilities, Pool, House Level/Style, Total rooms above ground.


**Analysis**
Saleprice has been performed in log_saleprice. Most of the analysis result coincide with the buyers interest. There are a few features that does not correlate with log_saleprice but have a weak or no correlation. This include lot area (+0.37 correlation), Central Air (+0.38 correlation), number of fireplace (+0.48 correlation), bedroom (+0.17 correlation), Utilities (+0.036 correlation), Fence (-0.15 correlation) and pool (+0.043 correlation).

This could possibly due to the possibility of lacking of representative samples (eg fence, pool), the over representative of the sample with favouring a particular option (eg central air, utilities) and another better representative/parameter features (eg lot area with above grade living area, number of fireplace with fireplace quality, and bedroom with total rooms above grade).

With these features, we will then proceed to perform the modeling.

----
### Data Preprocessing and Modeling & Analysis Summary

The features were then perform train_test_split at train_size of 0.7, standard_scalar and perform the modeling.
The analysis is as followed.

**Analysis 1: Ridge Regression performing the best with an R score of about 0.866.**

The cross_val_score across these models are approximately the same with a score of 0.866 with Ridge Regression performing better than the rest with 0.868. The Ridge Regressions functions by penalizing the size of the weights by introducing a regularization parameter. This model is useful for our analysis in which we have many predictors that may have a high degree of Multicollinearity between each other [[1]](https://towardsdatascience.com/from-linear-regression-to-ridge-regression-the-lasso-and-the-elastic-net-4eaecaf5f7e6) However, using ridge regression requires standard scaling and select an optimal alpha (hyperparameter) for its peak performance. 

Despite that, this regression is preferable over Lasso Regression as the cross validation score for this model is the highest out of the three and it helps to shrinks our regression coefficients closer to zero by making our model simpler.

**Analysis 2: Train score, test score and cross val score are similar with about 87-88%, which indicates the model prediction is reliable.**

Both our train and test score is about 0.87, which indicates that 87% of the variability in sale price can be explained by the x predictors in our model. From this, we obtain the coefficients. The top 5 features the sales price is heavily affected by the changes are above grade living area, overall quality of the house, ms zoning at Residential Low Density, the interaction of total basement area and the quality of the basement as well as the fireplace quality. 

The two zones that the sale price is affected mainly at Residential Low Density and Floating Village Residential with each unit change resulting in 6.5% and 3.3% increase in sale price compared to the other zones with 0-1.8% increase. Interestingly, kitchen quality, garage area and heating quality plays a significant role in sale price that results in an increase of 3.2-4.8% in sale price for every unit increase. Meanwhile, there are a few features that has very low correlation such as basement quality and full bath that has very minimum changes to sales prices. This is in particular with full bath that causes a decrease in sale price by 1% for each unit increase despite full bath is one of the top consideration of a buyer.

Therefore based on the model prediction, It is best to have a minimum amount of full bath with a focus on the features of the residential zome of low density or floating village, above grade living area, overall quality of the house, fireplace as well as the basement area and its quality. As these features heavily impacted the change in saleprice of the house.

---

### Conclusions

In conclusion, we have addressed our problem statement stated in the beginning, we have first performed data cleaning for the datasets followed by exploring and visualizing the data using exploratory data analysis. We then performed extensive feature engineering on various attributes that have resulted in an increase correlation to the house prices, such as total basement area and basement quality, garage area and garage finish, fireplace quality and heating quality. 

Based on the engineered data, we built three linear regression models: Linear Regression, Lasso and Ridge. The three models yield similar and stable results for the cross validation score. We have chosen Ridge as our model as the cross validation score for this model is the highest out of the three, and this model is useful for our analysis in which we have many predictors that may have a high degree of Multicollinearity between each other. The model accomplished an R2 score of approximately 89% on training sets and 90% on training-validation sets with an RMSE of 23395 and 22741 respectively. 

Using this model, we are able to identify the key features that are affecting the changes of the sale price of the house and utilise it to structure key features into future affordable and marketable homes.


---

### Recommendations

Overall, the top 5 features of the property that will affect the changes of the sale price are above grade living area, overall quality of the house, ms zoning at Residential Low Density, the interaction of total basement area and the quality of the basement as well as the fireplace quality. Interestingly, kitchen quality, garage area and heating quality were also identified as key features that enhance the sale price of the house. Meanwhile, it is best to have a minimum amount of full bath required in the house as it causes a decrease in sale price by 1% for each unit increase although full bath is listed as one of the buyer's feature consideration.

With these insights, our construction company in Ames Iowa is able to utilise this domain knowledge to build a affordable and desirable houses and advise homeowners to renovate their house focusing on key features that will increase their sale price, as well as estimate and predict the sale price given the features.

---
### References

[1] "Public and Private Lnads Percentages," 1991.[Online]. Available: https://www.summitpost.org/public-and-private-land-percentages-by-us-states/186111. [Accessed: Apr. 9. 2021].

[2] "Ames," *Niche.com.Inc*, 2021. [Online]. Available: https://www.niche.com/places-to-live/ames-story-ia/. [Accessed: Apr. 9. 2021]

[3] "QuickFacts Ames City, Iowa," *United States Census Bureau*, 2019. [Online]. Available: https://www.census.gov/quickfacts/amescityiowa. [Accessed: Apr. 6. 2021].

[4] "Housing in Ames, Iowa," *Sperling's Best Places*, [Online]. Available: https://www.bestplaces.net/housing/city/iowa/ames [Accessed: Apr. 9. 2021]

[5] "Iowa's affordable housing struggle," *The Gazette*, 2018.[Online]. Available: https://www.thegazette.com/regional-development/iowas-affordable-housing-struggle/. [Accessed: Apr. 9. 2021].

[6] "Ames, IA New Listings for Sale", *Movoto* 2021. [Online]. Available: https://www.movoto.com/ames-ia/new-7/. [Accessed: Apr. 6. 2021].

[7]"Homes in Ames, IA", *Kelller Williams* 2021. [Online]. Available: https://desmoines.homefront.com/homesearch/ames-ia. [Accessed: Apr. 6. 2021].

[8]"Real Estate & Homes For Sale", *Zillow* 2021. [Online]. Available: https://www.zillow.com/homes/. [Accessed: Apr. 6. 2021].

---

