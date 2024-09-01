from pulp import *
import random
import time
import matplotlib.pyplot as plt

# Set seed for reproducibility
random.seed(42)

# Define the recipes with their eligibility
recipes_f1_only = list(range(1, 30))
recipes_f1_f2 = list(range(30, 50))
recipes_f2_only = list(range(50, 90))
recipes_f3_only = list(range(90, 101))

def generate_order_recipes(eligible_recipes, max_recipes=4):
    return random.sample(eligible_recipes, random.randint(1, min(max_recipes, len(eligible_recipes))))

def generate_orders_for_day(total_orders, real_proportion, simulated_proportion, existing_real_orders=None):
    orders = []
    f1_f3_target = int(0.3 * total_orders)
    f2_f3_target = int(0.6 * total_orders)
    f1_f2_f3_target = int(0.1 * total_orders)
    
    total_real_orders = int(real_proportion * total_orders)
    
    # Create a set of all available IDs
    available_ids = set(range(1, total_orders + 1))
    
    if existing_real_orders:
        orders.extend([order.copy() for order in existing_real_orders])
        # Remove the IDs of existing orders from available_ids
        for order in orders:
            available_ids.remove(order['id'])
    
    # Generate new orders
    while len(orders) < total_orders:
        if len([o for o in orders if set(o['eligible_factories']) == {'F1', 'F2', 'F3'}]) < f1_f2_f3_target:
            recipe_ids = generate_order_recipes(recipes_f1_f2)
            eligible_factories = ['F1', 'F2', 'F3']
        elif len([o for o in orders if set(o['eligible_factories']) == {'F1', 'F3'}]) < f1_f3_target - f1_f2_f3_target:
            recipe_ids = generate_order_recipes(recipes_f1_only)
            eligible_factories = ['F1', 'F3']
        elif len([o for o in orders if set(o['eligible_factories']) == {'F2', 'F3'}]) < f2_f3_target - f1_f2_f3_target:
            recipe_ids = generate_order_recipes(recipes_f2_only)
            eligible_factories = ['F2', 'F3']
        else:
            f3_recipe = random.choice(recipes_f3_only)
            all_recipes = recipes_f1_only + recipes_f1_f2 + recipes_f2_only + recipes_f3_only
            remaining_recipes = random.sample(all_recipes, min(3, random.randint(0, 3)))
            recipe_ids = [f3_recipe] + remaining_recipes
            random.shuffle(recipe_ids)
            eligible_factories = ['F3']
        
        new_order = {
            'recipe_ids': recipe_ids,
            'eligible_factories': eligible_factories,
            'is_real': len([o for o in orders if o['is_real']]) < total_real_orders
        }
        
        # Assign a new ID from the available IDs
        new_id = available_ids.pop()
        new_order['id'] = new_id
        
        orders.append(new_order)
    
    # Adjust real/simulated order proportions
    real_order_count = sum(1 for order in orders if order['is_real'])
    if real_order_count < total_real_orders:
        for order in random.sample([o for o in orders if not o['is_real']], total_real_orders - real_order_count):
            order['is_real'] = True
    elif real_order_count > total_real_orders:
        for order in random.sample([o for o in orders if o['is_real']], real_order_count - total_real_orders):
            order['is_real'] = False
    
    # Shuffle orders to mix real and simulated orders
    random.shuffle(orders)
    
    return orders

def allocate_orders(orders, factory_capacities):
    allocation = {factory: [] for factory in factory_capacities}
    remaining_orders = orders.copy()

    # Allocate to F1 first, prioritizing F1,F3 and F1,F2,F3 orders
    f1_eligible = sorted([order for order in remaining_orders if 'F1' in order['eligible_factories']],
                         key=lambda x: len(x['eligible_factories']))  # Prioritize F1,F3 over F1,F2,F3
    allocation['F1'] = f1_eligible[:factory_capacities['F1']]
    remaining_orders = [order for order in remaining_orders if order not in allocation['F1']]

    # Allocate to F2, using remaining F1,F2,F3 orders and F2,F3 orders
    f2_eligible = [order for order in remaining_orders if 'F2' in order['eligible_factories']]
    allocation['F2'] = f2_eligible[:factory_capacities['F2']]
    remaining_orders = [order for order in remaining_orders if order not in allocation['F2']]

    # Allocate remaining to F3
    allocation['F3'] = remaining_orders

    return allocation

def get_recipe_counts(orders, allocation):
    recipe_counts = {f: {} for f in ['F1', 'F2', 'F3']}
    for factory, order_ids in allocation.items():
        for order_id in order_ids:
            if isinstance(order_id, dict):
                order = order_id
            else:
                order = next((o for o in orders if o['id'] == order_id), None)
            
            if order is None:
                continue
            
            for recipe in order['recipe_ids']:
                if recipe not in recipe_counts[factory]:
                    recipe_counts[factory][recipe] = 0
                recipe_counts[factory][recipe] += 1
    return recipe_counts

def calculate_wmape_site(allocation_t_minus_1, allocation_t, orders_t_minus_1, orders_t):
    recipe_counts_t_minus_1 = get_recipe_counts(orders_t_minus_1, allocation_t_minus_1)
    recipe_counts_t = get_recipe_counts(orders_t, allocation_t)
    
    total_abs_diff = 0
    total_items_t = 0
    
    all_recipes = set()
    for counts in recipe_counts_t_minus_1.values():
        all_recipes.update(counts.keys())
    for counts in recipe_counts_t.values():
        all_recipes.update(counts.keys())
    
    for recipe_id in all_recipes:
        for factory in ['F1', 'F2', 'F3']:
            t_minus_1_count = recipe_counts_t_minus_1[factory].get(recipe_id, 0)
            t_count = recipe_counts_t[factory].get(recipe_id, 0)
            total_abs_diff += abs(t_count - t_minus_1_count)
            total_items_t += t_count
    
    return total_abs_diff / total_items_t if total_items_t > 0 else 0

def calculate_wmape_global(allocation_t_minus_1, allocation_t, orders_t_minus_1, orders_t):
    recipe_counts_t_minus_1 = {}
    recipe_counts_t = {}
    
    for factory in allocation_t_minus_1:
        for order in allocation_t_minus_1[factory]:
            if isinstance(order, dict):
                order_recipes = order['recipe_ids']
            else:
                order = next((o for o in orders_t_minus_1 if o['id'] == order), None)
                if order is None:
                    continue
                order_recipes = order['recipe_ids']
            
            for recipe_id in order_recipes:
                recipe_counts_t_minus_1[recipe_id] = recipe_counts_t_minus_1.get(recipe_id, 0) + 1
    
    total_t_items = 0
    for factory in allocation_t:
        for order_id in allocation_t[factory]:
            order = next((o for o in orders_t if o['id'] == order_id), None)
            if order is None:
                continue
            for recipe_id in order['recipe_ids']:
                recipe_counts_t[recipe_id] = recipe_counts_t.get(recipe_id, 0) + 1
                total_t_items += 1
    
    all_recipes = set(recipe_counts_t_minus_1.keys()) | set(recipe_counts_t.keys())
    
    total_abs_diff = sum(abs(recipe_counts_t_minus_1.get(recipe_id, 0) - recipe_counts_t.get(recipe_id, 0)) for recipe_id in all_recipes)
    
    return total_abs_diff / total_t_items if total_t_items > 0 else 0

def run_optimization(total_orders):
    start_time_total = time.time()

    # Update factory capacities based on total_orders
    F1_cap = int(0.25 * total_orders)
    F2_cap = int(0.5 * total_orders)
    factory_capacities = {
        'F1': F1_cap,
        'F2': F2_cap,
        'F3': float('inf')
    }

    # Generate orders for day t-1 (LD12)
    orders_t_minus_1 = generate_orders_for_day(total_orders, 0.46, 0.54)

    # Extract real orders from day t-1
    real_orders_t_minus_1 = [order for order in orders_t_minus_1 if order['is_real']]

    # Generate orders for day t (LD11), including existing real orders
    orders_t = generate_orders_for_day(total_orders, 0.52, 0.48, existing_real_orders=real_orders_t_minus_1)

    # Perform allocation for day t-1
    allocation_t_minus_1 = allocate_orders(orders_t_minus_1, factory_capacities)

    # Start timing the optimization
    start_time_optimization = time.time()

    # Create the model
    prob = LpProblem("Order_Allocation", LpMinimize)

    # Create binary variables for order allocation
    x = LpVariable.dicts("allocation", [(o['id'], f) for o in orders_t for f in factory_capacities.keys()], cat='Binary')

    # Add constraints to ensure each order is allocated to exactly one factory
    for o in orders_t:
        prob += lpSum([x[o['id'], f] for f in factory_capacities.keys()]) == 1

    # Add constraints to ensure orders are only allocated to eligible factories
    for o in orders_t:
        for f in factory_capacities.keys():
            if f not in o['eligible_factories']:
                prob += x[o['id'], f] == 0

    # Add capacity constraints for F1 and F2
    prob += lpSum([x[o['id'], 'F1'] for o in orders_t]) == factory_capacities['F1']
    prob += lpSum([x[o['id'], 'F2'] for o in orders_t]) == factory_capacities['F2']

    # Calculate recipe counts for day t based on allocation
    recipe_counts_t = {}
    for f in factory_capacities.keys():
        for o in orders_t:
            for r in o['recipe_ids']:
                if r not in recipe_counts_t:
                    recipe_counts_t[r] = {}
                if f not in recipe_counts_t[r]:
                    recipe_counts_t[r][f] = 0
                recipe_counts_t[r][f] += x[o['id'], f]

    # Calculate absolute differences for WMAPE site
    abs_diffs = []
    recipe_counts_t_minus_1 = get_recipe_counts(orders_t_minus_1, allocation_t_minus_1)
    for r in set(recipe_counts_t.keys()) | set(sum((list(counts.keys()) for counts in recipe_counts_t_minus_1.values()), [])):
        for f in factory_capacities.keys():
            t_minus_1_count = recipe_counts_t_minus_1[f].get(r, 0)
            t_count = recipe_counts_t.get(r, {}).get(f, 0)
            
            # Create new variables for positive and negative differences
            pos_diff = LpVariable(f"pos_diff_{r}_{f}", lowBound=0)
            neg_diff = LpVariable(f"neg_diff_{r}_{f}", lowBound=0)
            
            # Add constraints to define the differences
            prob += t_count - t_minus_1_count == pos_diff - neg_diff
            
            abs_diffs.append(pos_diff + neg_diff)

    # Set the objective to minimize the sum of absolute differences (WMAPE site numerator)
    prob += lpSum(abs_diffs)

    # Solve the problem
    prob.solve()

    # End timing the optimization
    end_time_optimization = time.time()
    optimization_time = end_time_optimization - start_time_optimization

    # Extract the optimal allocation
    optimal_allocation = {f: [] for f in factory_capacities.keys()}
    for o in orders_t:
        for f in factory_capacities.keys():
            if value(x[o['id'], f]) == 1:
                optimal_allocation[f].append(o['id'])

    # Calculate WMAPE site and global
    optimized_wmape_site = calculate_wmape_site(allocation_t_minus_1, optimal_allocation, orders_t_minus_1, orders_t)
    wmape_global = calculate_wmape_global(allocation_t_minus_1, optimal_allocation, orders_t_minus_1, orders_t)

    end_time_total = time.time()
    total_execution_time = end_time_total - start_time_total

    return optimized_wmape_site, wmape_global, optimization_time, total_execution_time

# Run scalability test
order_quantities = [10000, 30000, 50000, 70000, 100000]
results = []

for quantity in order_quantities:
    print(f"Running optimization for {quantity} orders...")
    optimized_wmape_site, wmape_global, optimization_time, total_execution_time = run_optimization(quantity)
    results.append((quantity, optimization_time, optimized_wmape_site, wmape_global))
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print("--------------------")

# Print final results
print("\nOrder quantity | Optimization time (s) | WMAPE site (B&B) | WMAPE global")
print("---------------|------------------------|-------------------|-------------")
for result in results:
    quantity, opt_time, wmape_site, wmape_global = result
    print(f"{quantity:14d} | {opt_time:21.2f} | {wmape_site:17.3f} | {wmape_global:12.3f}")

# Plotting time
plt.figure(figsize=(10, 5))
plt.plot([r[0] for r in results], [r[1] for r in results], marker='o')
plt.title('Optimization time vs Order quantity')
plt.xlabel('Order quantity')
plt.ylabel('Optimization time (seconds)')
plt.tight_layout()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot([r[0] for r in results], [r[2] for r in results], marker='s', label='WMAPE site (B&B)')
plt.plot([r[0] for r in results], [r[3] for r in results], marker='^', label='WMAPE global')
plt.title('WMAPE vs Order quantity')
plt.xlabel('Order quantity')
plt.ylabel('WMAPE')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()