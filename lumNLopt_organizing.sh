# Step 1: Create the main lumNLopt folder and missing subfolders
mkdir -p lumNLopt/utilities
mkdir -p lumNLopt/adjoint_methods
mkdir -p lumNLopt/materials

# Create __init__.py files
touch lumNLopt/__init__.py
touch lumNLopt/utilities/__init__.py
touch lumNLopt/adjoint_methods/__init__.py
touch lumNLopt/materials/__init__.py

# Step 2: Move the existing folders into lumNLopt
mv figures_of_merit lumNLopt/
mv geometries lumNLopt/
mv lumerical_methods lumNLopt/
mv optimizers lumNLopt/

# Step 3: Move individual files to their proper locations
# Move core file
mv optimization.py lumNLopt/

# Move utilities
mv gradients.py lumNLopt/utilities/
mv materials.py lumNLopt/utilities/
mv fields.py lumNLopt/utilities/
mv wavelengths.py lumNLopt/utilities/
mv base_script.py lumNLopt/utilities/
mv edge.py lumNLopt/utilities/
mv simulation.py lumNLopt/utilities/
mv plotter.py lumNLopt/utilities/
mv scipy_wrappers.py lumNLopt/utilities/
mv load_lumerical_scripts.py lumNLopt/utilities/

# Step 4: Add missing __init__.py files to existing folders
touch lumNLopt/figures_of_merit/__init__.py
touch lumNLopt/geometries/__init__.py
touch lumNLopt/lumerical_methods/__init__.py
touch lumNLopt/optimizers/__init__.py

# Step 5: Check the result
find lumNLopt -name "*.py" | head -20

# Step 6: Commit and push the organized structure
git add .
git commit -m "Complete lumNLopt directory organization

- Created main lumNLopt package structure  
- Moved all files to appropriate subdirectories
- Added __init__.py files for Python packages
- Ready for FDTD adjoint enhancements with 3,600 parameters"
git push
