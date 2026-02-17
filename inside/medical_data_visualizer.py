import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# Question 1: Import the data from medical_examination.csv
# ============================================================================
df = pd.read_csv('data/medical_examination.csv')


# Question 2: Add an 'overweight' column to the data
# ============================================================================

# 1. Convert height from cm to meters
height_meter = df['height'] / 100
# 2. Calculate BMI: weight / (height_in_meters ** 2)
BMI = df['weight'] / (height_meter ** 2)
# 3. Create 'overweight' column: 1 if BMI > 25, else 0
df['overweight'] = (BMI > 25).astype(int)


# Question 3: Normalize the data
# ============================================================================

# Normalize cholesterol
df['cholesterol'] = np.where(df["cholesterol"] == 1, 0, 1)

# Normalize glucose
df['gluc'] = np.where(df["gluc"] == 1, 0, 1)


# Question 4-8: Draw the Categorical Plot
# ============================================================================
def draw_cat_plot():
    
    # Question 5: Create DataFrame for cat plot using pd.melt
    # ========================================================================

    # GOAL: Convert wide format to long format
    df_cat = pd.melt(
        df,
        id_vars = ['cardio'],
        value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )
    
    
    # Question 6: Group and reformat the data
    # ========================================================================

    # GOAL: Count how many people have each value for each variable
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Question 7: Create the catplot
    # ========================================================================

    # SEABORN CATPLOT PARAMETERS:
    fig = sns.catplot(
         x='variable',
         y='total',
         hue='value',
         col='cardio',
         data=df_cat,
         kind='bar'
         )
    
    
    # Question 8: Get the figure for output
    # ========================================================================
    fig.set_axis_labels("Variable", "Count")
    fig.set_titles("Cardio = {col_name}")
    fig = fig.fig
    fig.savefig('catplot.png')
    return fig


# Question 9-15: Draw the Heat Map
# ============================================================================
def draw_heat_map():
    
    # Question 10: Clean the data
    # ========================================================================

    # GOAL: Filter out incorrect/impossible data
    df_heat = df[
         (df['ap_lo'] <= df['ap_hi']) &
         (df['height'] >= df['height'].quantile(0.025)) &
         (df['height'] <= df['height'].quantile(0.975)) &
         (df['weight'] >= df['weight'].quantile(0.025)) &
         (df['weight'] <= df['weight'].quantile(0.975))

         ]
    
    
    # Question 11: Calculate the correlation matrix
    # ========================================================================

    # CORRELATION MATRIX:
    corr = df_heat.corr()
    
    
    # Question 12: Generate a mask for the upper triangle
    # ========================================================================
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    
    # Question 13: Set up the matplotlib figure
    # ========================================================================    
    fig, ax = plt.subplots(figsize = (12, 10))
    
    
    # Question 14: Plot the correlation matrix using seaborn heatmap
    # ========================================================================    
    sns.heatmap(
         corr,
         mask=mask,
         annot=True,
         fmt='.1f',
         square=True,
         linewidths=0.5,
         cbar_kws={'shrink': 0.5},
         ax=ax
     )
    fig.savefig('heatmap.png')
    return fig


# HELPER FUNCTIONS
# ============================================================================

def get_data_summary():
    
    print("=" * 60)
    print("MEDICAL DATA SUMMARY")
    print("=" * 60)
    print(f"\nTotal patients: {len(df)}")
    print(f"\nCardiovascular disease distribution:")
    print(df['cardio'].value_counts())
    print(f"\nAge statistics:")
    print(df['age'].describe())
    print(f"\nGender distribution:")
    print(df['sex'].value_counts())
    print("=" * 60)


def explain_correlations():
    
    print("""
    CORRELATION INTERPRETATION:
    
    Positive correlation (0 to 1):
      0.0 - 0.3: Weak positive relationship
      0.3 - 0.7: Moderate positive relationship
      0.7 - 1.0: Strong positive relationship
    
    Negative correlation (0 to -1):
      -0.3 - 0.0: Weak negative relationship
      -0.7 - -0.3: Moderate negative relationship
      -1.0 - -0.7: Strong negative relationship
    
    Example:
      If age and cardio have correlation of 0.5:
      → As age increases, cardio disease moderately increases
    """)


# TESTING YOUR CODE
# ============================================================================

if __name__ == '__main__':
    
    print("Loading medical data...")
    
    # Test data loading
    if df is not None:
        print(f"✓ Data loaded: {len(df)} patients")
        get_data_summary()
    else:
        print("✗ Data not loaded yet. Complete INSTRUCTION 1.")
    
    # Test plots
    print("\nGenerating categorical plot...")
    try:
        draw_cat_plot()
        print("✓ Categorical plot saved as 'catplot.png'")
    except Exception as e:
        print(f"✗ Error in categorical plot: {e}")
    
    print("\nGenerating correlation heatmap...")
    try:
        draw_heat_map()
        print("✓ Heatmap saved as 'heatmap.png'")
    except Exception as e:
        print(f"✗ Error in heatmap: {e}")

