import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ===============================================
# ==== Define the Input and Output Variables ====
# ===============================================

# define inputs of defect detection
hole_defect = ctrl.Antecedent(np.arange(0, 101, 0.1), "Hole Defect")
object_defect = ctrl.Antecedent(np.arange(0, 101, 0.1), "Object")
oil_spot = ctrl.Antecedent(np.arange(0, 101, 0.1), "Oil Spot")
thred_error = ctrl.Antecedent(np.arange(0, 101, 0.1), "Thread Error")
knot = ctrl.Antecedent(np.arange(0, 101, 0.1), "Knot")

# define input of color detection
color = ctrl.Antecedent(np.arange(0, 11, 1), "Color")

# define input of fiber composition
fiber_comp = ctrl.Antecedent(np.arange(0, 101, 0.1), "Fiber Composition")

# define output of fuzzy inference system
recyclability = ctrl.Consequent(np.arange(0, 101,0.1), "Recyclability")
reusability = ctrl.Consequent(np.arange(0, 101, 0.1), "Reusability")
downgrade = ctrl.Consequent(np.arange(0, 101, 0.1), "Downgrade")

# ==========================================
# ======= Define Membership Functions =======
# ==========================================

# For Hole defect
hole_defect["minor"] = fuzz.gaussmf(hole_defect.universe, 0, 10)
hole_defect["major"] = fuzz.gaussmf(hole_defect.universe, 50, 10)
hole_defect["critical"] = fuzz.gaussmf(hole_defect.universe, 100, 10)

# for object detection
object_defect["minor"] = fuzz.gaussmf(object_defect.universe, 0, 10)
object_defect["major"] = fuzz.gaussmf(object_defect.universe, 50, 10)
object_defect["critical"] = fuzz.gaussmf(object_defect.universe, 100, 10)

# for oil spot detection
oil_spot["minor"] = fuzz.gaussmf(oil_spot.universe, 0, 10)
oil_spot["major"] = fuzz.gaussmf(oil_spot.universe, 50, 10)
oil_spot["critical"] = fuzz.gaussmf(oil_spot.universe, 100, 10)

# for thread error detection
thred_error["minor"] = fuzz.gaussmf(thred_error.universe, 0, 10)
thred_error["major"] = fuzz.gaussmf(thred_error.universe, 50, 10)
thred_error["critical"] = fuzz.gaussmf(thred_error.universe, 100, 10)

# for knot detection
knot["minor"] = fuzz.gaussmf(knot.universe, 0, 10)
knot["major"] = fuzz.gaussmf(knot.universe, 50, 10)
knot["critical"] = fuzz.gaussmf(knot.universe, 100, 10)

# for color detection
color["low"] = fuzz.trimf(color.universe, [0, 0, 5])
color["medium"] = fuzz.trimf(color.universe, [0, 5, 10])
color["high"] = fuzz.trimf(color.universe, [5, 10, 10])

# for fiber composition
fiber_comp["natural"] = fuzz.trimf(fiber_comp.universe, [0, 0, 50])
fiber_comp["blend"] = fuzz.trimf(fiber_comp.universe, [0, 50, 100])
fiber_comp["synthetic"] = fuzz.trimf(fiber_comp.universe, [50, 100, 100])

# for recyclability
recyclability["low"] = fuzz.trimf(recyclability.universe, [0, 0, 50])
recyclability["medium"] = fuzz.trimf(recyclability.universe, [0, 50, 100])
recyclability["high"] = fuzz.trimf(recyclability.universe, [50, 100, 100])

# for reusability
reusability["low"] = fuzz.trimf(reusability.universe, [0, 0, 50])
reusability["medium"] = fuzz.trimf(reusability.universe, [0, 50, 100])
reusability["high"] = fuzz.trimf(reusability.universe, [50, 100, 100])

# for downgrade
downgrade["low"] = fuzz.trimf(downgrade.universe, [0, 0, 50])
downgrade["medium"] = fuzz.trimf(downgrade.universe, [0, 50, 100])
downgrade["high"] = fuzz.trimf(downgrade.universe, [50, 100, 100])


def calculate_sorting(damage_percentages, color_class_counter, fiber_composition=50):
    """
    Runs the fuzzy inference system with the given data.
    """
    # ======================================
    # ======== Define the Fuzzy Rules ========
    # ======================================

    # --- Reusability Rules ---
    rule1 = ctrl.Rule(hole_defect['minor'] & object_defect['minor'] & oil_spot['minor'] & thred_error['minor'] & knot['minor'], reusability['high'])
    rule2 = ctrl.Rule(hole_defect['major'] | object_defect['major'] | thred_error['major'] | knot['major'], reusability['medium'])
    rule3 = ctrl.Rule(hole_defect['critical'] | object_defect['critical'] | oil_spot['critical'] | thred_error['critical'] | knot['critical'], reusability['low'])

    # --- Recyclability Rules ---
    rule4 = ctrl.Rule((fiber_comp['natural'] | fiber_comp['synthetic']) & color['high'] & oil_spot['minor'], recyclability['high'])
    rule5 = ctrl.Rule((fiber_comp['natural'] | fiber_comp['synthetic']) & (color['medium'] | oil_spot['major']), recyclability['medium'])
    rule6 = ctrl.Rule(fiber_comp['blend'] | color['low'] | oil_spot['critical'], recyclability['low'])

    # --- Downgrade Rules ---
    rule7 = ctrl.Rule(reusability['low'] & recyclability['low'], downgrade['high'])
    rule8 = ctrl.Rule(reusability['low'] & recyclability['medium'], downgrade['medium'])
    rule9 = ctrl.Rule(reusability['high'] | recyclability['high'], downgrade['low'])

    # =========================================
    # ====== Create the Control System ========
    # =========================================

    sorting_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    sorting_system = ctrl.ControlSystemSimulation(sorting_ctrl)

    # =========================================
    # ========= Run the Simulation ============
    # =========================================

    # --- Provide crisp input values ---
    sorting_system.input['Hole Defect'] = damage_percentages[0] * 100
    sorting_system.input['Object'] = damage_percentages[1] * 100
    sorting_system.input['Oil Spot'] = damage_percentages[2] * 100
    sorting_system.input['Thread Error'] = damage_percentages[3] * 100
    sorting_system.input['Knot'] = damage_percentages[4] * 100
    
    color_index = np.argmax(color_class_counter)
    if (color_index == 0): # blend
        sorting_system.input['Color'] = 1
    elif (color_index == 1): # dark_shade
        sorting_system.input['Color'] = 5 
    elif (color_index == 2): # light_shade
        sorting_system.input['Color'] = 8
    elif (color_index == 3): # white
        sorting_system.input['Color'] = 10

    sorting_system.input['Fiber Composition'] = fiber_composition

    # --- Compute the result ---
    sorting_system.compute()

    # --- Return the output values ---
    reusability_score = sorting_system.output['Reusability']
    recyclability_score = sorting_system.output['Recyclability']
    downgrade_score = sorting_system.output['Downgrade']
    
    return reusability_score, recyclability_score, downgrade_score




