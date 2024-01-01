#!/usr/bin/env python
# coding: utf-8

# ## Comprehensive USA Customer Transaction Analysis And Predictive Modeling
# 
# #### Introduction
# Welcome to the "Comprehensive USA Customer Transaction Analysis And Predictive Modeling" project! Here, I take you on a journey through the intricate world of customer transactions, focusing specifically on the USA market. As the sole creator of this project, I've seamlessly merged tables, preprocessed data, engineered features, conducted exploratory data analysis (EDA), visualized patterns, and employed machine learning techniques for prediction. This project is a solo endeavor, showcasing a
# harmonious blend of analytical techniques to derive meaningful insights and predictions from a rich dataset.

# ### Project Overview
# #### 1. Data Merging
# 
# We kick off our journey by merging tables with common columns, bringing together diverse aspects of customer transactions.
# 
# #### 2. Data Preprocessing
# 
# Ensuring data quality and consistency through cleaning and preparation for downstream analysis.
# 
# #### 3. Feature Engineering
# 
# Crafting new features to extract additional insights and improve the predictive power of our models.
# 
# #### 4. Exploratory Data Analysis (EDA)
# 
# 1.Unveiling the hidden stories within the data through a series of visualizations and analyses:
# 
# 2.Visualizing the correlation matrix of numerical features.
# 
# 3.Exploring the relationship between Avg_Price and GST through scatter plots.
# 
# 4.Analyzing the distribution of categorical features (Gender, Location, Coupon_Status).
# 
# 5.Investigating total sales over time.
# 
# 6.Examining total transactions, total amount, and average discount percentage by product category and month.
# 
# 7.Unveiling the monthly distribution of transactions, variation in total sales, and total amount by product category and gender.
# 
# #### 5. Machine Learning
# 
# Utilizing machine learning algorithms for predictive modeling:
# 
# 1.Predicting Total_Amount using Linear Regression.
# 
# 2.Predicting Coupon_Status using Logistic Regression.
# 
# 3.Predicting Product_Category using RandomForestClassifier.
# 
# #### 6. Neural Network
# 
# Implementing a Neural Network for precise Product Category prediction.

# ### Column Details
# CustomerID: Unique identifier for each customer.
# 
# Gender: Gender of the customer (M for male in the provided examples).
# 
# Location: Location or city where the transaction took place (e.g., Chicago).
# 
# Tenure_Months: Number of months the customer has been associated with the service.
# 
# Transaction_ID: Unique identifier for each transaction.
# 
# Transaction_Date: Date of the transaction (e.g., 2019-01-01).
# 
# Product_SKU: Stock-keeping unit for the product.
# 
# Product_Description: Detailed description of the product purchased.
# 
# Product_Category: Category to which the product belongs (e.g., Nest-USA).
# 
# Quantity: Quantity of the product purchased in the transaction.
# 
# Avg_Price: Average price of the product.
# 
# Delivery_Charges: Charges for delivering the product.
# 
# Transaction_Month: Month in which the transaction occurred (e.g., Jan for January).
# 
# Coupon_Status: Status of any coupon used in the transaction (e.g., Used).
# 
# Coupon_Code: Code associated with the used coupon (e.g., ELEC10).
# 
# Discount_pct: Percentage of discount applied to the transaction (e.g., 10.0).
# 

# In[ ]:





# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # 1. Data Merging:- Merging Tables with Common Columns
# Merging tables with common columns is a data manipulation task that typically falls under the broader categories of "data manipulation" or "data preprocessing." Specifically, it is part of the process known as "data merging" or "data joining." This task involves combining two or more tables based on a common column or key to create a unified dataset.

# ### 1.customer table

# In[2]:



customer=pd.read_csv('C:\\Users\\rushi\\Downloads\\Current Project\\Customer.csv')
customer.head(2)


# In[3]:


print(customer.info())
print(customer.isnull().sum())


# ### 2.online_sales

# In[4]:


online_sales=pd.read_csv('C:\\Users\\rushi\\Downloads\\Current Project\\Online_Sales.csv')
online_sales.head(2)


# In[5]:


print(online_sales.info())
print(online_sales.isnull().sum())


# In[6]:


# Convert 'Transaction_Date' to datetime format
online_sales['Transaction_Date'] = pd.to_datetime(online_sales['Transaction_Date'])# the pd.to_datetime() function is used to convert the 'Transaction_Date' column to a datetime format.
#Create a new column 'Transaction_Month' with month information
 #the dt.strftime('%b') method is used to extract the month abbreviation (e.g., 'Jan', 'Feb') and create a new column 'Transaction_Month'.
online_sales.insert(9, 'Transaction_Month', online_sales['Transaction_Date'].dt.strftime('%b'))
online_sales.head(2)


# ## 1.Merge Online Sales And Customer Table on CustomerID
# 

# In[7]:


#Merged Table 
consumer = pd.merge(customer, online_sales, on='CustomerID', how='inner')
consumer.head(2)


# ### 3.discount_coupon table 

# In[8]:


discount_coupon=pd.read_csv('C:\\Users\\rushi\\Downloads\\Current Project\\Discount_Coupon.csv')
discount_coupon.head(2)


# In[9]:


print(discount_coupon.info())
print(discount_coupon.isnull().sum())


# ## 2.Merged  consumer and discount_coupon Table on 'Month',  & 'Product_Category'

# In[10]:


consumer1 = pd.merge(consumer, discount_coupon, 
                     left_on=['Transaction_Month', 'Product_Category'],
                     right_on=['Month', 'Product_Category'],
                     how='left')


# In[11]:


# Drop the redundant 'Month' column from the merged DataFrame
consumer1 = consumer1.drop(columns=['Month'])

# Display the result
print(consumer1[['CustomerID','Transaction_Month', 'Product_Category', 'Coupon_Code', 'Discount_pct']].head(2))


# In[12]:


#Merge Table 
consumer1.head(2)


# ###### Find Null Values in consumer1 table

# In[13]:


len(discount_coupon['Product_Category'].unique())


# In[14]:


len(consumer1['Product_Category'].unique())


# In[15]:


consumer1.isnull().sum()


# In[16]:


# Find unique values in 'Product_Category' column of discount_coupen
discount_coupon_categories = set(discount_coupon['Product_Category'].unique())

consumer1_categories = set(consumer1['Product_Category'].unique())

# Find unique values that are different between the two sets
different_categories = discount_coupon_categories.symmetric_difference(consumer1_categories)

print(different_categories)


# In[17]:


#because of {'Notebooks', 'Google', 'Backpacks', 'More Bags', 'Fun'} different categories in consumer1 table -
#it shows null values 400 


# In[ ]:





# ### 4.taxamount 

# In[18]:


taxamount=pd.read_csv('C:\\Users\\rushi\\Downloads\\Current Project\\Tax_amount.csv')
taxamount.head(2)


# In[19]:


print(taxamount.info())
print(taxamount.isnull().sum())


# ## 3.Merge consumer1 and taxamount  on Product_Category

# In[20]:


sales= pd.merge(consumer1,taxamount, on='Product_Category', how='left')


# ## Final Table Created - sales

# In[21]:


sales.head()


# In[ ]:





# # 2.Data Preprocessing 
# Data preprocessing refers to the set of procedures and techniques applied to raw data before analysis. Its purpose is to clean, transform, and organize the data, making it suitable for further exploration, modeling, or analysis. Data preprocessing aims to enhance the quality and relevance of the data, addressing issues such as missing values, outliers, and inconsistencies, to ensure more accurate and meaningful results in subsequent data-driven tasks.

# In[22]:


sales.info()


# In[23]:


categories_to_extract = ['Google', 'Notebooks', 'Backpacks', 'More Bags', 'Fun']

# Use boolean indexing to filter rows based on the specified categories
filtered_sales = sales[sales['Product_Category'].isin(categories_to_extract)]

# Display the result
print(filtered_sales[['Product_Category','Transaction_Month', 'Coupon_Status',]])


# #### Fill null values

# In[24]:


sales.isnull().sum()


# In[25]:


sales['Coupon_Code'].fillna('SALE10',inplace=True)
sales['Discount_pct'].fillna(10,inplace=True)


# In[26]:


sales[['Coupon_Code','Discount_pct']].isnull().sum()


# #### Define the correct data types for each column
# 

# In[27]:


data_types = {
    'CustomerID': int,
    'Gender': str,
    'Location': str,
    'Tenure_Months': int,
    'Transaction_ID': int,
    'Transaction_Date': 'datetime64[ns]',
    'Product_SKU': str,
    'Product_Description': str,
    'Product_Category': str,
    'Quantity': int,
    'Avg_Price': float,
    'Delivery_Charges': float,
    'Transaction_Month': str,
    'Coupon_Status': str,
    'Coupon_Code': str,
    'Discount_pct': float,
}

# Apply the correct data types to each column
sales = sales.astype(data_types)
# Convert GST Column in integer 
sales['GST'] = sales['GST'].str.rstrip('%').astype(int)

# Display the result
print(sales.dtypes)


# In[28]:


sales.head()


# In[ ]:





# # 3.Feature engineering
# Feature engineering is the process of selecting, modifying, or creating new features from raw data to enhance the performance and effectiveness of machine learning models, aiding in better model understanding, accuracy, and generalization.

# #### Create New Feature(Column)- Total Amount 

# In[29]:


# Calculate the 'Total_Amount' column
#1st Calculating  on columns ( (Quantity multiply by Avg_Price) + Delivery_Charges
sales['Total_Amount'] = (sales['Quantity'] * sales['Avg_Price']) + sales['Delivery_Charges']

#2nd if Coupon_Status Is Used then applying Dicount on total amount
sales.loc[sales['Coupon_Status'] == 'Used', 'Total_Amount'] *= (1 - sales['Discount_pct'] / 100)
 
#3rd applying GST based on Product_Category on Total Amount 
sales['Total_Amount'] *= (1 + sales['GST'] / 100)


# In[30]:


sales['Total_Amount'] = sales['Total_Amount'].round(2)


# In[31]:


#final result of Total Amount
sales.head(5)


# In[ ]:





# # 4.Exploratory Data Analysis. 
# EDA It is an approach to analyzing and summarizing key characteristics of a dataset in order to gain insights, identify patterns, and understand the underlying structure. EDA involves using various statistical and visual techniques to explore the data. The primary goal of EDA is to provide a comprehensive overview of the dataset.

# In[32]:


# Visualize the distribution of numerical features
#num_cols = ['Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'Discount_pct', 'GST', 'Total_Amount']
#sns.pairplot(sales[num_cols])
#plt.show()


# ## 1.Visualize the correlation matrix of numerical features.

# In[33]:


# Visualize the correlation matrix of numerical features
num_cols = ['Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'Discount_pct', 'GST', 'Total_Amount']
correlation_matrix = sales[num_cols].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Positive Correlation 
# 
# 1.Quantity has a high positive correlation with Total_Amount (Quantity increase - Total amount increase )
# 
# 2.Quantity has a high positive correlation with Total_Amount (Quantity increase - Total amount increase )
# 
# 3.Discount_pct has a high negative correlation with Total_Amount (Discount Increase - Total Amount Decrease )

# Negative Correlation 
# 
# 1.highly negative correlation between GST (Goods and Services Tax) and Avg_Price could suggest that when the average price 
# of products increases, the amount of GST tends to decrease
# 
# 2.higher-priced items are more likely to have discounts or be excluded from GST. In some regions, certain products may be exempt from GST or have reduced tax rates.

# ## 2.Scatter plot between Avg_Price and GST.

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot between Avg_Price and GST
plt.figure(figsize=(8,4))
sns.scatterplot(x='Avg_Price', y='GST', data=sales)
plt.title('Scatter Plot: Avg_Price vs GST')
plt.xlabel('Average Price')
plt.ylabel('GST')
plt.show()


# The scatter plot indicates an inverse relationship between Avg_Price and GST, suggesting that as average prices rise, GST tends to decrease. This may imply a pricing strategy where higher-priced items incur lower GST percentages or vice versa.

# ## 3.Distribution of categorical features Gender ,Location , Coupon_Status.
# 

# In[35]:


# Distribution of categorical features
cat_cols = ['Gender', 'Location', 'Coupon_Status']
for col in cat_cols:
    plt.figure(figsize=(12,6))
    sns.countplot(x=col, data=sales)
    plt.title(f'Distribution of {col}')
    plt.show()


# ### 1.Gender Distribution:
# In terms of gender distribution, the dataset shows a higher representation of women compared to men.
# Specifically, the total sales attributed to women exceed 30,000, while the total sales associated with men are approximately in the range of 20,000 to 21,000.
# This suggests that women have a higher contribution to the overall sales in the dataset.
# ### 2.Location Distribution:
# The dataset reveals a notable disparity in customer distribution across various locations, with a substantial number of customers hailing from Chicago and California, each contributing to more than 15,000 transactions. In contrast, customer representation is comparatively lower in New Jersey and Washington DC, where transactions are fewer than 5,000, indicating a lower customer presence in these regions.
# ### 3.Coupon Status Distribution:
# The distribution of coupon statuses in the dataset shows that a significant number of transactions fall into the "Used" category, ranging between 17,000 and 18,000. In contrast, the "Not Used" category has a lower count, with transactions in the range of 7,000 to 8,000. Additionally, the data indicates a high volume of transactions where customers clicked on coupons, surpassing 25,000 transactions.Clicked suggests customer engagement with the coupon, but it does not guarantee that the coupon was applied or resulted in a completed transaction. It provides insights into customer interest or curiosity about the offered discounts or promotions.

# ## 4.Total sales over time.

# In[36]:


sales['Transaction_Date'] = pd.to_datetime(sales['Transaction_Date'])
sales['Transaction_YearMonth'] = sales['Transaction_Date'].dt.to_period('M')
monthly_sales = sales.groupby('Transaction_YearMonth')['Total_Amount'].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title('Total Sales Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales Amount(USD)')
plt.show()


# ### Total Sales Over Time:
# 1.The total sales exhibit varying patterns over the months. Notably, from September to January, there is a consistent upward trend, indicating higher total sales amounts during this period.
# 
# 2.In particular, December stands out with a significant spike in total sales, exceeding 550,000(USD) This surge is likely influenced by the festive Christmas season, suggesting increased consumer interest and purchasing activity during the holiday period.
# Additionally, July also reflects a notable peak in total sales, approaching $450,000, suggesting another period of heightened consumer engagement.
# 
# 3.Conversely, May and June show a decrease in total sales, falling below $375,000, indicating a potential dip in consumer spending during these months.

# ##  5.Total Transactions by Product Category.

# In[37]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Product_Category', data=sales)
plt.title('Count of Transactions by Product Category')
plt.xlabel('Product Category')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()


# ## 6.Total Amount by Product Category.

# In[1]:


'''
Display values above the bars

for p in ax.patches:
    
    ax.annotate(f'${p.get_height():,.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

for p in ax.patches:: This is a loop that iterates through each patch (bar or rectangle) in the ax object. In Matplotlib, a bar chart is typically created using patches.
ax.annotate(...): This function is used to add text annotations to the plot.
f'${p.get_height():,.2f}': This part creates a formatted string that represents the height of the current bar (p.get_height()) in currency format (${:,.2f}). It displays the height with two decimal places and adds a dollar sign.
(p.get_x() + p.get_width() / 2., p.get_height()): Specifies the coordinates where the annotation will be placed. It's positioned at the center of the bar. p.get_x() gives the x-coordinate of the left side of the bar, and p.get_width() / 2 offsets it to the center. p.get_height() provides the y-coordinate of the top of the bar.
ha='center', va='center': These parameters set the horizontal and vertical alignment of the text to the center.
xytext=(0, 10): Specifies the offset of the text from the specified coordinates. In this case, the text is placed 10 points above the center of the bar.
textcoords='offset points': Indicates that the xytext values are specified in points.
fontsize=8: Sets the font size of the annotation text to 8 points.
In summary, this code snippet goes through each bar in the plot and adds an annotation above the center of the bar, displaying the height of the bar in currency format with two decimal places. The annotation is offset 10 points above the bar's center.
'''


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'sales' is your DataFrame
plt.figure(figsize=(20,8))

# Calculate total amount by product category
total_amount_by_category = sales.groupby('Product_Category')['Total_Amount'].sum().reset_index()

# Sort the DataFrame by total amount in descending order
total_amount_by_category = total_amount_by_category.sort_values(by='Total_Amount', ascending=False)

# Create a bar plot
ax = sns.barplot(x='Product_Category', y='Total_Amount', data=total_amount_by_category, palette='viridis')

# Display values above the bars
for p in ax.patches:
    ax.annotate(f'${p.get_height():,.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

plt.title('Total Amount by Product Category (Descending Order)')
plt.xlabel('Product Category')
plt.ylabel('Total Amount (USD)')
plt.xticks(rotation=45)
plt.show()


# In[39]:


# Calculate total amount by each product category
total_amount_by_category = sales.groupby('Product_Category')['Total_Amount'].sum().reset_index()

# Sort the DataFrame by total amount in descending order
total_amount_by_category = total_amount_by_category.sort_values(by='Total_Amount', ascending=False)

# Print the total amount for each product category
for index, row in total_amount_by_category.iterrows():
    print(f"{row['Product_Category']}: ${row['Total_Amount']:,.2f}")    


# The total sales amount by product category reveals significant contributions from Nest-USA, dominating with 2,724,198.68(USD), followed by strong performances from Apparel (846,172.54(USD)) and Nest (520,208.93(USD)). Other notable categories include Office(384,322.43(USD)), Drinkware(278,013.59(USD)), and Bags(196,328.31(USD)), contributing to the overall sales success. On the contrary, categories like More Bags and Android have relatively lower sales figures.Waze(12,913.67(USD)), Google (12,910.79(USD)),Backpacks(10,638.25(USD)),Accessories(10,294.80(USD)),Bottles(10,121.82(USD)),Fun(9,044.47(USD)),Housewares(7,198.50(USD)),More Bags: (3,963.61(USD)),Android(1,122.82(USD))  have relatively lower sales figures. This breakdown provides insights into the distribution of sales across diverse product categories, highlighting key performers in the dataset.

# ## 7.Heatmap of Average Discount Percentage by Product Category and Month.

# In[40]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'sales' is your DataFrame
# Convert 'Transaction_Date' to datetime
sales['Transaction_Date'] = pd.to_datetime(sales['Transaction_Date'])

# Extract month from 'Transaction_Date'
sales['Transaction_Month'] = sales['Transaction_Date'].dt.month_name()

# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Calculate average discount percentage by product category and month
avg_discount_by_category_month = sales.groupby(['Product_Category', 'Transaction_Month'])['Discount_pct'].mean().reset_index()

# Pivot the data to create a matrix for the heatmap
heatmap_data = avg_discount_by_category_month.pivot(index='Product_Category', columns='Transaction_Month', values='Discount_pct')
heatmap_data = heatmap_data[month_order]  # Reorder columns based on month_order

# Plot the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f", linewidths=.5)
plt.title('Average Discount Percentage by Product Category and Month')
plt.xlabel('Month')
plt.ylabel('Product Category')
plt.show()


# Monthly Variation in Discount Percentage by Product Category: Peaks in March, June, September, and December.

# ## 8.Monthly Distribution of Transactions.

# In[41]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Transaction_Month', data=sales, order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], palette='viridis')
plt.title('Count of Transactions by Month')
plt.xlabel('Month')
plt.ylabel('Transaction Count')
plt.show()


# ##### Distribution of Monthly Transactions: Peaks in July and August, Medium Activity in May and December, and Similar Levels in Other Months, Low transaction in February.

# ## 9.Monthly Variation in Total Sales by Product Category.

# In[42]:


# Calculate total sales by product category and month
total_sales_by_category_month = sales.groupby(['Transaction_Month', 'Product_Category'])['Total_Amount'].sum().reset_index()

# Sort the DataFrame by total sales in descending order
total_sales_by_category_month = total_sales_by_category_month.sort_values(by=['Transaction_Month', 'Total_Amount'], ascending=[True, False])

# Plot the results with a dark color palette
plt.figure(figsize=(14, 8))
sns.barplot(x='Product_Category', y='Total_Amount', hue='Transaction_Month', data=total_sales_by_category_month, palette='bright')
plt.title('Total Sales by Product Category and Month')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Amount (USD)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Month', bbox_to_anchor=(1, 1))
plt.show()


# 1.Nest USA:
# 
# Highs: Nov, Dec, Jan
# Lows: May, Sept
# Moderate Activity: Other months
# 
# 2.Apparel:
# 
# Highs: Apr, July
# Moderate Activity: Mar, May, Aug
# Decline Post-August: Until Feb
# 
# 3.Office:
# 
# Highs: Jan, March, April
# Lows: Sept, Oct, Dec
# Moderate Activity: Other months
# 
# 4.Nest:
# 
# Sales Surge: Aug to Dec
# Peaks: Nov, Dec
# Limited Activity: Jan to July
# 
# 5.Drinkware:
# 
# Highs: March, Apr, Oct
# Lows: Nov, Dec
# Moderate Activity: Other months
# 
# 6.Notebook and Journal:
# 
# Highs: Feb, Apr, Aug
# Lower Activity: Other months
# 
# 8.Bags, Lifestyle, Headgear, Nest-Canada:
# 
# Consistently Low Sales: All months (10,000 to 40,000 USD range)
# 
# 9.Other Products:
# 
# Generally Low Sales: Below 10,000 USD consistently

# ## 10.Total Amount by Product Category and Gender Heatmap.

# In[43]:


# List of product categories
product_categories = ['Nest-USA', 'Office', 'Apparel', 'Bags', 'Drinkware', 'Lifestyle',
                       'Waze', 'Headgear', 'Fun', 'Google', 'Notebooks & Journals',
                       'Backpacks', 'Nest-Canada', 'Housewares', 'Bottles', 'Nest',
                       'Android', 'Accessories', 'Gift Cards', 'More Bags']

# Calculate total amount by product category, gender, and month
total_amount_by_category_gender = sales.groupby(['Product_Category', 'Gender'])['Total_Amount'].sum().reset_index()

# Filter data for the specified product categories
total_amount_by_category_gender = total_amount_by_category_gender[total_amount_by_category_gender['Product_Category'].isin(product_categories)]

# Pivot the data to create a matrix for the heatmap
heatmap_data = total_amount_by_category_gender.pivot(index='Product_Category', columns='Gender', values='Total_Amount')

# Plot the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".2f", linewidths=.5)
plt.title('Total Amount by Product Category and Gender')
plt.xlabel('Gender')
plt.ylabel('Product Category')
plt.show()


# ##### Total Amount by Product Category and Gender Heatmap: High Men's and Women's Spending on NEST USA, Apparel, Nest, and Office; Moderate for Bags, Lifestyle, Headgear, and Nest-Canada; Low for Other Categories. Note: Men's Spending is Generally Lower Compared to Women's Spending on these Products."

# ## 11.Top 20 Customers by Total Transaction Amount and Location.

# In[44]:


import pandas as pd

# Assuming 'sales' is your DataFrame
# You may need to adapt column names based on your actual DataFrame structure

# Group by CustomerID and calculate total transaction amount
customer_transaction_amount = sales.groupby('CustomerID')['Total_Amount'].sum().reset_index()

# Sort by transaction amount in descending order and select the top 20
top_20_customers = customer_transaction_amount.sort_values(by='Total_Amount', ascending=False).head(20)

# Merge with the original DataFrame to get Customer ID, Transaction Amount, and Country
top_20_customers_info = pd.merge(top_20_customers, sales[['CustomerID', 'Location']], on='CustomerID').drop_duplicates()

# Set index from 1 to 20
top_20_customers_info.index = range(1, 21)
print("Top 20 Customers")
# Display the top 20 customers in table format
print(top_20_customers_info[['CustomerID', 'Total_Amount', 'Location']])


# In[45]:


plt.figure(figsize=(12, 6))
sns.countplot(x='Location', data=top_20_customers_info, palette='viridis')
plt.title('Distribution of Locations for Top 20 Customers')
plt.xlabel('Location')
plt.ylabel('Count')
plt.show()


# ##### Top 20 Customers by Total Transaction Amount and Location: Chicago (9), California (7), New Jersey (2), New York (1), Washington DC (1)

# ## 12.GST by Product Category and Location.

# In[46]:


filtered_sales = sales[sales['GST'] < 6]


# In[47]:


# Create a bar plot for GST by product category and location
plt.figure(figsize=(14, 8))
sns.barplot(x='Product_Category', y='GST', hue='Location', data=sales, ci=None, palette='pastel')
plt.title('GST by Product Category and Location')
plt.xlabel('Product Category')
plt.ylabel('GST')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Location', bbox_to_anchor=(1, 1))
plt.show()


# Uniform GST Across Locations: Highest GST (18%) for Apparel, Bags, More Bags, Drinkware, Lifestyle, and Fun. GST Rates: Notebook, Headgear, Bottles, Nest Gift (6%); Remaining Products (10%)

# ## 13.Distribution of Tenure Months by Customer.

# In[48]:


print("The number of unique customers is:-",sales['CustomerID'].nunique())  #nunique() method to count the number of unique customer IDs in the 'CustomerID' column 
# Group by CustomerID and calculate the maximum Tenure_Months for each customer
customer_tenure_info = sales.groupby('CustomerID')['Tenure_Months'].max().reset_index()

# Create equal-width bins for Tenure_Months
bins = pd.cut(customer_tenure_info['Tenure_Months'], bins=[0, 10, 20, 30, 40, 50])

# Plot the distribution using a countplot
plt.figure(figsize=(12, 6))
sns.countplot(x=bins, palette='viridis')
plt.title('Distribution of Tenure Months by Customer')
plt.xlabel('Tenure Months Bin')
plt.ylabel('Count')
plt.show()


# The distribution of customer tenure reveals interesting patterns. The majority of customers fall into the 20-30 and 30-40 months tenure bins, indicating a concentration of long-term customers in these ranges. Additionally, the distribution appears relatively uniform for the remaining tenure bins, suggesting a consistent presence of customers across different tenure periods. This insight underscores the importance of customer retention, especially among those with longer tenures in the 20-40 months range
# 

# In[ ]:





# # 5.Machine Learning 
# Machine learning is a field of artificial intelligence that focuses on the development of algorithms and models that enable computers to learn from and make predictions or decisions based on data, without being explicitly programmed.

# In[49]:


df=sales.copy()


# ## 1.Total_Amount Predict Using Linear regression. 

# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[51]:


# Label encode categorical columns if needed
label_encoder = LabelEncoder()


# In[52]:


from sklearn.preprocessing import LabelEncoder

# Columns to encode
columns_to_encode = ['Gender', 'Location', 'Product_Category',"Transaction_Month","Coupon_Status"]

# Dictionary to store mappings
mappings = {}

# Iterate through colu
for column in columns_to_encode:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    
    # Print the mapping
    print(f"{column} Mapping:")
    column_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(column_mapping)
    
    # Store the mapping in the dictionary
    mappings[column] = column_mapping

# Display the updated DataFrame


# In[53]:



# Select features (X) and target variable (y)
features = ['Gender', 'Location','Transaction_Month','Product_Category', 'Quantity', 'Avg_Price','Discount_pct',"Coupon_Status"]
X = df[features]
y = df['Total_Amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
lin_model_ta = LinearRegression()

# Train the model
lin_model_ta.fit(X_train, y_train)

lin_model_ta.score(X_test,y_test)#model.score(X_test,y_test)


# In[54]:


X_train.head(2)


# In[55]:


lin_model_ta.predict([[1,0,3,16,100,120,0,2]])


# In[ ]:





# ## 2.Coupon_Status Prediction Using  LogisticRegression.
# 

# In[56]:


from sklearn.linear_model import LogisticRegression

features = ['Gender', 'Location', 'Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'Discount_pct', 'GST']
X = df[features]
y = df['Coupon_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
log_model = LogisticRegression()

# Train the model
log_model.fit(X_train, y_train)


# In[57]:


log_model.score(X_test,y_test)


# In[58]:


X_test.head(2)


# In[59]:


log_model.predict([[1,1,32,24,100,0,0,5]])


# ### 1.Apply RandomForestClassifier

# In[60]:


from sklearn.ensemble import RandomForestClassifier

R_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
R_model.fit(X_train, y_train)


# In[61]:


R_model.score(X_test,y_test)


# ##### RandomForestClassifier Score  - 0.42

# ### 2.Apply KNeighborsClassifier  

# In[62]:


from sklearn.neighbors import KNeighborsClassifier
K_model = KNeighborsClassifier(n_neighbors=5)
K_model.fit(X_train, y_train)


# In[63]:


K_model.score(X_test,y_test)


# ##### KNeighborsClassifier  Score 0.45

# ### AdaBoostClassifier with LogisticRegression

# In[64]:


from sklearn.ensemble import AdaBoostClassifier
base_estimator = LogisticRegression(solver='liblinear', multi_class='ovr')
adb=AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)


# In[65]:


cross_val_score(adb, X_train,y_train, cv=3)


# ##### Score - O.50

# In[ ]:





# ## 3.Predicting Product_Category using  RandomForestClassifier.

# In[66]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Assuming your data is stored in a DataFrame named 'df'
# You might need to preprocess your data, handle missing values, and encode categorical variables

# Select features (X) and target variable (y)
features = ['Gender', 'Location', 'Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'Discount_pct', 'GST']
X = df[features]
y = df['Product_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier model
R_model_pc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
R_model_pc.fit(X_train, y_train)

# Make predictions on the test set
predictions = R_model_pc.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Print classification report for more detailed evaluation
print(classification_report(y_test, predictions))


# In[67]:


R_model_pc.score(X_test,y_test)


# In[68]:


R_model_pc.score(X_train,y_train)


# In[69]:


R_model_pc.predict([[1,1,22,10,119,8,20,0]])


# In[ ]:





# # 6.Neural Network 
# A neural network is a computational model inspired by the structure and functioning of the human brain, composed of interconnected nodes (neurons) arranged in layers. It is used in machine learning to recognize patterns, make predictions, and perform tasks by learning from data through iterative training processes.

# ## Product Category Prediction with Neural Network.
# 1.This script demonstrates building and training a neural network for product category prediction.
# 
# 2.It uses features like Gender, Location, Tenure_Months, Quantity, Avg_Price, etc., to predict the product category.
# 
# 3.The model is trained using TensorFlow and Keras, and evaluated using accuracy and classification report.
# 
# 4.Code snippet taken from ChatGPT.
# 

# In[70]:


import tensorflow as tf 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[71]:


# Assuming your data is stored in a DataFrame named 'df'
# You might need to preprocess your data, handle missing values, etc.

# Specify the features and target variable
features = ['Gender', 'Location', 'Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'Discount_pct', 'GST']
target_variable = 'Product_Category'

# Extract features and target variable
X = df[features]
y = df[target_variable]

# Label encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[72]:


# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=len(features), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer with softmax for classification



# In[73]:


# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[74]:


# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)


# In[75]:


# Inverse transform the encoded predictions to get the original classes
y_pred_original = label_encoder.inverse_transform(y_pred_classes)


# In[76]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Accuracy: {accuracy}')


# In[77]:


X_test.head(2)


# ##### Pedict Product 

# In[79]:


new_input = [[1, 1, 25, 1, 24.99, 6.50, 30, 5]]

# Make predictions for the new input
new_input_predictions = model.predict(new_input)
new_input_class = new_input_predictions.argmax(axis=-1)[0]

# Inverse transform the encoded prediction to get the original product category
new_input_original_category = label_encoder.inverse_transform([new_input_class])[0]

# Print the predicted product category
print("Predicted Product Category for Input:", new_input_original_category)


# ##### Headgear: 10

# In[ ]:





# ### Product Category Prediction with Multiple Classifiers

# 1.Product Category Prediction with Multiple Classifiers (Code Commented due to Extended Execution Time)
# 2.This script demonstrates using RandomForest, Support Vector Machine (SVM), and k-Nearest Neighbors (kNN)
# 3.classifiers for product category prediction. Features like Gender, Location, Tenure_Months, Quantity, Avg_Price, etc., are used.
# 4.Cross-validation is performed to evaluate classifiers, but the actual training and testing code is commented to reduce execution time.
# 

# In[ ]:


'''
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Assuming your data is stored in a DataFrame named 'df'
# You might need to preprocess your data, handle missing values, and encode categorical variables

# Example of preprocessing for demonstration purposes
# Label encode categorical columns if needed

# Select features (X) and target variable (y)
features = ['Gender', 'Location', 'Tenure_Months', 'Quantity', 'Avg_Price', 'Delivery_Charges', 'Discount_pct', 'GST']
X = df[features]
y = df['Product_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
svm_classifier = SVC(kernel='linear', random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# List of classifiers
classifiers = [rf_classifier, svm_classifier, knn_classifier]

# Perform cross-validation and print accuracy scores for each classifier
for classifier in classifiers:
    scores = cross_val_score(classifier, X, y, cv=3)
    print(f"Accuracy for {classifier.__class__.__name__}: {scores.mean()}")

# If you want to check accuracy on a specific test set
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on test set for {classifier.__class__.__name__}: {accuracy}")

'''


# In[ ]:




