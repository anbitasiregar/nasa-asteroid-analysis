# Asteroid Predictive Analysis and Modeling

## Objective
This dataset comes from NASA’s NeoWs (Near Earth Object Web Service), a RESTful web service that details asteroid information near Earth. I'm going to be using this data to understand:
1. what features are most correlated with an asteroid being classified as hazardous or not
2. Finding the best classification model to accurately predict whether an asteroid is hazardous or not

## Business Problem
This analysis would be great for NASA and the aerospace and space exploration industry to use for planetary defense and risk management.
- Resource Allocation: Space organizations often have limited resources for monitoring and researching asteroids. This model could help prioritize which asteroids need the most immediate attention or study, allowing organizations to allocate resources more efficiently.
- Partnerships and Funding: This project could also be used to pitch for funding or collaborations with other organizations who are interested in asteroid risk management. Governments and other organizations could also use this project as a tool to provide transparent data about asteroids for monitoring or emergency preparedness plans.
- Public awareness: This project could be used to educate the masses on asteroids and could become an open-source tool for people to do their own research with. 

## Data Understanding
This dataset contains 4687 rows and has 40 features to describe asteroids. There were no null values to clean, but feature engineering and feature selection techniques were used to make the data more relevant for analysis and modeling.

Target feature: `Hazardous`

## Analysis & Results

### Insight #1
Through descriptive analysis, we found that hazardous asteroids, on average, are larger than non-hazardous asteroids. Larger asteroids pose a a greater potential for catastrophic damage if they were to impact Earth, making their early detection and monitoring a priority.

### Insight #2
Hazardous asteroids tend to come closer to Earth’s orbit than non-hazardous asteroids. This proximity increases the likelihood of close encounters, making such asteroids more dangerous.

### Insight #3
Orbit intersection distance is the most important feature to classify hazard type of an asteroid. This feature had the strongest correlation with the target variable and consistently influenced model performance.

### Winning model
**Random Tree Classifier** with an accuracy of **99.5%**

## Conclusions

### Recommendation #1:
Create an internal alerting system that would flag asteroids above a certain size threshold. This will help NASA to focus resources on assessing more credible threats.

### Recommendation #2:
Enhance monitoring systems for asteroids that come within a defined distance of Earth's orbit. This data and monitoring system can be integrated into planetary defense systems to improve real-time tracking and ensure preparedness for potential impacts.

### Recommendation #3:
Refine and tune predictive models to focus on this feature, along with others that show high predictive power like absolute magnitude and asteroid diameter. This focus can improve classification accuracy and reduce false negatives, ensuring that potentially hazardous asteroids are accurately flagged, minimizing the risk of missing critical threats.

### Next Steps
- Further Research: Explore additional asteroid features or collect new data to improve model predictions and understand asteroid behavior.
- Collaboration and Fundraising: Use this analysis to pitch for funding and secure partnerships with government agencies, private organizations, and international space programs to pool resources for planetary defense.
- Public Awareness Campaigns: Educate the public about asteroid risks and mitigation efforts. One idea is to develop an open-source tool for others to explore asteroid data and contribute to the field.
- Real-Time Application: Integrate the Random Forest model into real-time asteroid tracking systems and other decision-making pipelines to classify and prioritize asteroids dynamically.

## For More Information
See the full analysis in the Jupyter Notebook or review this presentation

Original data source: https://www.kaggle.com/datasets/lovishbansal123/nasa-asteroids-classification/data

## Repository Structure
```bash
├── data
├── images
├── notebooks
├── presentations
├── .gitignore
├── README.md
```