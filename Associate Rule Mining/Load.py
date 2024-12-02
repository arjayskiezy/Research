import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth

# Load the dataset
file_path = "./DATASET/ASM.csv"  # Adjust the extension if necessary
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("Dataset Preview:")
print(data.head())

# Ensure the dataset is preprocessed with one-hot encoding for categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
one_hot_encoded_data = pd.get_dummies(data, columns=categorical_columns)

# Split the dataset into training (80%) and testing (20%)
train_data, test_data = train_test_split(one_hot_encoded_data, test_size=0.2, random_state=42)

# Apply the Apriori algorithm on the training set
frequent_itemsets_train = apriori(train_data, min_support=0.2, use_colnames=True)

# Generate association rules from the frequent itemsets
rules_train = association_rules(frequent_itemsets_train, metric="lift", min_threshold=1.0)

# Evaluate the rules on the test set by recalculating support, confidence, and lift
frequent_itemsets_test = apriori(test_data, min_support=0.2, use_colnames=True)
rules_test = association_rules(frequent_itemsets_test, metric="lift", min_threshold=1.0)

# Visualization 1: Support vs Confidence (Training Set)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules_train, x="support", y="confidence", size="lift", hue="lift", palette="viridis", sizes=(50, 300))
plt.title("Support vs Confidence (Training Set)")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.legend(title="Lift")
plt.show()

# Visualization 2: Top 10 Rules by Lift (Training Set)
top_rules_train = rules_train.nlargest(10, 'lift')
plt.figure(figsize=(10, 6))
sns.barplot(x=top_rules_train['lift'], y=top_rules_train['consequents'].apply(lambda x: ', '.join(list(x))), palette="coolwarm")
plt.title("Top 10 Rules by Lift (Training Set)")
plt.xlabel("Lift")
plt.ylabel("Consequents")
plt.show()

# Visualization 3: Frequent Itemsets by Support (Training Set)
frequent_itemsets_train['itemsets'] = frequent_itemsets_train['itemsets'].apply(lambda x: ', '.join(list(x)))
top_itemsets_train = frequent_itemsets_train.nlargest(10, 'support')
plt.figure(figsize=(10, 6))
sns.barplot(x=top_itemsets_train['support'], y=top_itemsets_train['itemsets'], palette="Blues_d")
plt.title("Top 10 Frequent Itemsets by Support (Training Set)")
plt.xlabel("Support")
plt.ylabel("Itemsets")
plt.show()

# Validation: Check test set performance of rules
test_rules_evaluation = []
for _, rule in rules_train.iterrows():
    antecedents = set(rule['antecedents'])
    consequents = set(rule['consequents'])
    total_matches = 0
    consequent_matches = 0
    
    for _, row in test_data.iterrows():
        if antecedents.issubset(set(row[row == 1].index)):
            total_matches += 1
            if consequents.issubset(set(row[row == 1].index)):
                consequent_matches += 1
    
    support_test = total_matches / len(test_data)
    confidence_test = consequent_matches / total_matches if total_matches > 0 else 0
    lift_test = confidence_test / (support_test if support_test > 0 else 1)
    
    test_rules_evaluation.append((rule['antecedents'], rule['consequents'], support_test, confidence_test, lift_test))

# Create a DataFrame for test rules evaluation
test_rules_df = pd.DataFrame(test_rules_evaluation, columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])

# Display test set results
print("\nValidation Results on Test Set:")
print(test_rules_df)

# Visualization 4: Support vs Confidence (Test Set)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=test_rules_df, x="Support", y="Confidence", size="Lift", hue="Lift", palette="magma", sizes=(50, 300))
plt.title("Support vs Confidence (Test Set)")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.legend(title="Lift")
plt.show()
