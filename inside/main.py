import medical_data_visualizer
from unittest import main


# Test my visualizations
print("🏥 MEDIVISION PRO - Medical Data Visualizer")
print("=" * 60)

# Generate visualizations
print("\n📊 Generating categorical plot...")
medical_data_visualizer.draw_cat_plot()
print("✓ Saved as 'catplot.png'")

print("\n🔥 Generating correlation heatmap...")
medical_data_visualizer.draw_heat_map()
print("✓ Saved as 'heatmap.png'")
print("\n" + "=" * 60)

# Run unit tests
print("\n🧪 Running unit tests...")
try:
    main(module='test_module', exit=False, verbosity=2)
except:
    print("⚠️  No test_module.py found. Skipping tests.")
    print("💡 Tests are optional for the dashboard version.")
