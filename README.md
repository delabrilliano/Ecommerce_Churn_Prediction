# **Exploratory Data Analysis on Crimes in Boston Dataset**
![head](https://raw.githubusercontent.com/delabrilliano/Ecommerce_Churn_Prediction/main/image/ecommerce.png)


**By: Delabrilliano Ismail**
<hr>

## **Introduction**
Many of the early-stage e-commerce business focuses on "Customer Acquisition", which is the act of gaining new customers.

Acquisition is particularly important for early-stage e-commerce looking to grow their customer base.  But it’s not a sustainable way to grow the company revenue in the long term.

In order to be sustainable in the long-term, an e-commerce business must also focus on customer retention. Knowing when a customer will churn, can help a company in general to retain their customer better for a long-term profit.

Fundamentally, churn occurs when a customer stops consuming from a company. A high churn rate equals to a low retention rate.  Churn affects the size of a company customer base and has a big impact on the company's customer lifetime value.

<hr>

## **Problem Statement**
Maintaining a churn-rate is crucial for an e-commerce business long-term profit. However, not all company have a system that can detect which of their customer that will churn. This situation can have a bad consequences for the company, when they give a benefit/promotion to a non-churning consumer and let the other customer churned. In this case, the company will compound an expense and lost a potential revenue from the churned consumer. Or, if a company want to play safe, they can give the benefit/promotion for all of their customer base. However, it will require a quite large of expenditure without any certainty that the benefit was given to the right target.

<hr>

## **Goals**

Based on the introduction and the problem statement, the aims for this project is to develop a machine learning model that can classify on which customer of a company in general and e-commerce company specifically, that will churn on not churn.

By having a system that can classify which customer will churn, an e-commerce company can give a benefit/promotion with a clearer target, and this will reduce their expense compared to when a company give the benefit/promotion to all of their customer base. And this will also have an impact on the company's customer retention rate, and in return, the company will have a more sustainable revenue in the future.

In short, this project is aims to develop a machine learning model, that can be used by marketing/sales/any department related to benefit/promotion, that can classify which customer will churn and not churn, that can help a company to maintain their customer retention rate to generate more sustainable revenue.

<hr>

## **Metrics**

- Class 0 = Non-Churn (Negative)
- Class 1 = Churn (Positive) 

_Type 1 error: False Positive_

This error will increase the company expense by giving benefit and promotion to the non-churn customer and ignore the churning customer.

_Type 2 Error: False Negative_

This error will make a company to overlook their churning customer without giving them any benefit/promotion to maintain their retention.

Based on the consequences, we will be focusing on both errors type. We want to make sure that the model can detect a churning customer as many as possible and ensure that the company give the benefit and promotion to the right customer (churned customer). We also have to notes that the data is imbalance (which will be shown later in the EDA). So the metrics that we will use is **F1 Score**.

Precision and Recall are the two building blocks of the F1 score. The goal of the F1 score is to combine the precision and recall metrics into a single metric. At the same time, the F1 score has been designed to work well on imbalanced data. [(Source)](https://towardsdatascience.com/the-f1-score-bec2bbc38aa6)

**F1 Score Formula**

\begin{equation}
F1 Score (+) = 2 *  \frac{Recall (+) * Precision (+)}{Recall (+) + Precision (+)}
\end{equation}

<hr>

## **Analytics Approach**

Based on the Problem statement and goals, we will be conducting the _**Prescriptive Analytics**_ where we will predict which customer will churn, make a decision based on the prediction (giving benefit and promotion), and analyze how these decision impact the business.

<hr>

## **Data Understanding**

**Notes:**

The dataset was given and the features are already pre-selected by _Purwadhika Digital Technology School_ as part of the exams for the Data Science and Machine Learning bootcamp programs. The csv file of the dataset is attached on this repository.

**Column Description**

| Attribute | Description |
| --- | --- |
| Tenure | Tenure of customer in organization |
| WarehouseToHome | Distance in between warehouse to home of customer |
| NumberOfDeviceRegistered | Total number of deceives is registered on particular customer |
| PreferedOrderCat | Preferred order category of customer in last month |
| SatisfactionScore | Satisfactory score of customer on service |
| MaritalStatus | Marital status of customer |
| NumberOfAddress | Total number of address added on particular customer |
| Complain | Any complaint has been raised in last month |
| DaySinceLastOrder | Day Since last order by customer |
| CashbackAmount | Average cashback in last month |
| Churn | Indicate the if the customer churned (1) or not (0) |

[(Source)](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

<hr>

## **Best Model Classification Report**

![ClassReport](https://raw.githubusercontent.com/delabrilliano/Ecommerce_Churn_Prediction/main/image/classreport.png)

<hr>

## **Conclusion**

### **Model**

#### **Recommended Model**

From the classification report, we can conclude that our best model and chosen model is the XGBoost with SMOTE technique and tuned hyperparameters.

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples. [(Source)](https://xgboost.readthedocs.io/en/stable/)

Whereas Synthetic Minority Oversampling Technique (SMOTE) is a statistical technique for increasing the number of cases in the dataset in a balanced way. The component works by generating new instances from existing minority classes that is supplied as input. The implementation of SMOTE does not change the number of majority classes. [(Source)](https://docs.microsoft.com/en-us/azure/machine-learning/component-reference/smote)

From the classification report, we can summarize that if the model is implemented, we can detect 87% of all the churning customer (recall score) and 78% of our model customer churning prediction is precise (precision score)

<hr>

### **Before-After**

#### **Before implementing ML models:**

Let's say we are an e-commerce company in USA.

According to [Invesp](https://www.invespcro.com/blog/global-online-retail-spending/) (Digital Consulting Firm), the average revenue per online shopper on e-commerce business in USA is $1,804.

Using our last confusion matrix on test dataset, let's say we currently have 789 customers (117 + 18 + 33 + 621).


- 789 customer at the beginning of a month
- 135 customer will churn (117+18)
- Revenue from each customer = $1,804
- Revenue = 789 * 1804 = $1.423.356

- Marketing Cost = 20% of Revenue (Average marketing cost for e-commerce) _*Source:_ [Boldist](https://boldist.co/marketing-strategy/ecommerce-digital-marketing-budget/)
- Marketing Cost = 20% * $1.423.356 = $284.671
- Average Customer Acquisiton Cost for e-commerce (retail) = $10 _*Source:_ [Propeller](https://www.propellercrm.com/blog/customer-acquisition-cost)

Let's say the cost for retaining a customer is the same with the cost for acquiring new customer:
- Customer Retention Cost = $10

Based on these data, the company with no system to detect a churning customer, will give all the customer base a benefit/promotion from the marketing budget. And spend the rest on acquiring new customer.

- Spent on customer retention: 789 * $10 = $7.890
- Spend the rest on new customer acquisition = $284.671 - $7.890 = $276.781
- New customer acquired = $276.781/10 = 27.678
- Customer for next month = 27.678 + 789 = 28.467 total customer
- Potential revenue for next month = 28.467 * $1.804 = **$51.354.468** 

#### **After implementing ML models:**

- 789 customer at the beginning of a month
- 135 customer will churn (117 True Positive + 18 False Negative)
- 150 customer detected will churn (117 True Positive + 33 False Positive)
- Revenue from each customer = $1,804
- Revenue = 789 * 1804 = $1.423.356

- Marketing Cost = 20% of Revenue
- Marketing Cost = 20% * $1.423.356 = $284.671
- Average Customer Acquisiton Cost for e-commerce (retail) = $10
- Customer Retention Cost = $10

After implementing ML models, the company will no longer give all the customer a benefit/promotion. Instead, they will only give the benefit/promotion to the detected customer that will churn.

- Spent on customer retention: 150 * $10 = $1.500
- Spend the rest on new customer acquisition = $284.671 - $1.500 = $283.171
- Churned customer = 33 (False Positive)
- New customer acquired = $283.171/10 = 28.317
- Customer for next month = 28.317 + 789 - 33 = 29.073 total customer
- Potential revenue for next month = 29.073 * $1.804 = **$52.447.632**

<hr>

We can see the potential revenue differences from implementing the ML models on our hypothetical case above. from the hypothetical case above, By implementing the ML models, the company can gain up to **1 million USD** ($ 1.093.164 to be exact) of potential revenue on the next month from the higher number of acquired customers.

<hr>

### **Limitation and Recommendation**

#### **Project Limitation**

The model was built on a dataset that was already pre-selected and limited, so the author realize there will be a limitation for the models/project in which the prediction will be less accurate on a certain condition. The limitation of this project are:

- Only limited amount of features used in this model

- Due to the outlier handling, there is a possibility that this model will got the prediction wrong when the data considered as outlier. The outlier criteria on each columns is:

| Features | Outlier |
| --- | --- |
| Tenure | Less than -19 / More than 37 year |
| WarehouseToHome | Less than -9 / More than 39 km|
| NumberOfDeviceRegistered | Less than 2 / More than 5 device |
| PreferedOrderCat | Category outside of 'Laptop & Accessory', 'Mobile', 'Fashion', 'Others', 'Mobile Phone', 'Grocery'|
| SatisfactionScore | Outside of scale from 1 to 5 |
| MaritalStatus | Status outside of 'Single', 'Married', 'Divorced' |
| NumberOfAddress | Less than -4 / more than 12 address |
| Complain | less than -1,5 / more than 2,5 complain |
| DaySinceLastOrder | less than -5,5 / more than 14,5 days |
| CashbackAmount | Less than 71,375 / more than 269,575 |

#### **Implementation Recommendation**

This project/machine learning model is recommended to be implemented by marketing/sales/any department related to benefit/promotion. The department can implement it in end of month of each month, to calculate or forecast the expected revenue on the next month. This model also can be implemented whenever the management or BODs want to measure their churn rate.

#### **Future Recommendation**

To improve this project/machine learning models, future projects can considers:
- Adding more features that is related to the target (customer Churn), such as Age, Redeemed vouchers/promo, Gender, etc.
- Use other algorithm such as SVM or LGBM and try other feature engineering such as scalling.
- Use other method to fill missing value / handle outlier

<hr>

_**Get in Touch:**_

| [Linkedin](https://www.linkedin.com/in/delabrilliano-ismail-05758715a/) | [Github](https://github.com/delabrilliano) | [Tableau](https://public.tableau.com/app/profile/delabrilliano.ismail)
<hr>
