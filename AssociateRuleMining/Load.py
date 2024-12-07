import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import association_rules, fpgrowth

# Step 1: Load the dataset
file_path = './DATASET/ASM.csv'
data = pd.read_csv(file_path)

# Step 2: Take a 50% random sample of the data
sampled_data = data.sample(frac=0.5, random_state=42)  # Set random_state for reproducibility

# Step 3: Split the sampled data into 80% training and 20% testing
train_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42)

# Verify the splits
print("Original Data Size:", len(data))
print("Sampled Data Size (50%):", len(sampled_data))
print("Training Data Size (80% of sampled):", len(train_data))
print("Testing Data Size (20% of sampled):", len(test_data))

# Step 4: Data Preprocessing for Eclat and FP-Growth
def preprocess_data(data):
    # Pivot the data to create a transactional matrix
    transactions = data.groupby(['STORECODE', 'BRD'])['QTY'].sum().unstack().reset_index().fillna(0)
    transactions.set_index('STORECODE', inplace=True)
    # Convert to boolean for memory efficiency
    return transactions > 0  # Binary conversion to boolean type

# Prepare training and testing datasets
basket_train = preprocess_data(train_data)
basket_test = preprocess_data(test_data)

# Step 5: Implement Eclat Algorithm
def eclat(data, min_support=0.05, max_itemset_size=3):
    """Simplified Eclat Algorithm for frequent itemset mining."""
    itemsets = {}
    num_transactions = len(data)

    # Step 1: Generate 1-itemsets
    for col in data.columns:
        support = data[col].sum() / num_transactions
        if support >= min_support:
            itemsets[frozenset([col])] = support

    k = 2
    while True:
        # Stop if max itemset size is reached
        if max_itemset_size and k > max_itemset_size:
            break

        items = list(itemsets.keys())
        new_itemsets = {}

        # Sequential computation to save resources
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                candidate = items[i].union(items[j])
                if len(candidate) == k:
                    support = (data[list(candidate)].all(axis=1).sum()) / num_transactions
                    if support >= min_support:
                        new_itemsets[candidate] = support

        if not new_itemsets:
            break
        itemsets.update(new_itemsets)
        k += 1

    return pd.DataFrame([(list(k), v) for k, v in itemsets.items()], columns=['itemset', 'support'])

# Run Eclat on training data
frequent_itemsets_eclat = eclat(basket_train, min_support=0.1, max_itemset_size=3)
print("Eclat Frequent Itemsets (Training Data):")
print(frequent_itemsets_eclat)

# Export Eclat results to CSV
frequent_itemsets_eclat.to_csv('./Eclat_Frequent_Itemsets.csv', index=False)

# Step 6: Implement FP-Growth Algorithm
frequent_itemsets_fpgrowth = fpgrowth(basket_train, min_support=0.1, use_colnames=True)
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1)

print("\nFP-Growth Frequent Itemsets (Training Data):")
print(frequent_itemsets_fpgrowth)

print("\nFP-Growth Rules (Training Data):")
print(rules_fpgrowth)

# Export FP-Growth results to CSV
frequent_itemsets_fpgrowth.to_csv('./FPGrowth_Frequent_Itemsets.csv', index=False)
rules_fpgrowth.to_csv('./FPGrowth_Rules.csv', index=False)

# Step 7: Testing Data Analysis (Optional)
frequent_itemsets_test_fpgrowth = fpgrowth(basket_test, min_support=0.1, use_colnames=True)
print("\nFP-Growth Frequent Itemsets (Testing Data):")
print(frequent_itemsets_test_fpgrowth)
