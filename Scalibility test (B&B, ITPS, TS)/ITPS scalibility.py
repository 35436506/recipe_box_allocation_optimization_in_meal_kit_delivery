import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Define the recipes with their eligibility
recipes_f1_only = list(range(1, 30))
recipes_f1_f2 = list(range(30, 50))
recipes_f2_only = list(range(50, 90))
recipes_f3_only = list(range(90, 101))

# Global variables that will be set in run_allocation_process
total_orders = None
factory_capacities = None

def generate_order_recipes(eligible_recipes, max_recipes=4):
    return random.sample(eligible_recipes, random.randint(1, min(max_recipes, len(eligible_recipes))))

def generate_orders_for_day(real_proportion, simulated_proportion, existing_real_orders=None):
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

def get_recipe_counts(allocation, by_factory=False):
    if by_factory:
        recipe_counts = {factory: {} for factory in allocation}
        for factory, orders in allocation.items():
            for order in orders:
                for recipe_id in order['recipe_ids']:
                    recipe_counts[factory][recipe_id] = recipe_counts[factory].get(recipe_id, 0) + 1
    else:
        recipe_counts = {}
        for factory, orders in allocation.items():
            for order in orders:
                for recipe_id in order['recipe_ids']:
                    recipe_counts[recipe_id] = recipe_counts.get(recipe_id, 0) + 1
    return recipe_counts

def calculate_wmape_site(allocation_t_minus_1, allocation_t):
    recipe_counts_t_minus_1 = get_recipe_counts(allocation_t_minus_1, by_factory=True)
    recipe_counts_t = get_recipe_counts(allocation_t, by_factory=True)
    
    all_recipes = set(recipe_counts_t_minus_1['F1'].keys()) | set(recipe_counts_t_minus_1['F2'].keys()) | set(recipe_counts_t_minus_1['F3'].keys()) | \
                  set(recipe_counts_t['F1'].keys()) | set(recipe_counts_t['F2'].keys()) | set(recipe_counts_t['F3'].keys())
    
    total_abs_diff = 0
    total_items_t = 0
    
    for recipe_id in sorted(all_recipes):
        for factory in ['F1', 'F2', 'F3']:
            t_minus_1_count = recipe_counts_t_minus_1[factory].get(recipe_id, 0)
            t_count = recipe_counts_t[factory].get(recipe_id, 0)
            abs_diff = abs(t_count - t_minus_1_count)
            total_abs_diff += abs_diff
            total_items_t += t_count
    
    wmape_site = total_abs_diff / total_items_t if total_items_t > 0 else float('inf')
    
    return wmape_site

def calculate_wmape_global(allocation_t_minus_1, allocation_t):
    recipe_counts_t_minus_1 = get_recipe_counts(allocation_t_minus_1)
    recipe_counts_t = get_recipe_counts(allocation_t)
    
    all_recipes = set(recipe_counts_t_minus_1.keys()) | set(recipe_counts_t.keys())
    
    total_abs_diff = 0
    total_t_items = sum(recipe_counts_t.values())
    
    for recipe_id in sorted(all_recipes):
        t_minus_1_count = recipe_counts_t_minus_1.get(recipe_id, 0)
        t_count = recipe_counts_t.get(recipe_id, 0)
        abs_diff = abs(t_minus_1_count - t_count)
        total_abs_diff += abs_diff
    
    wmape_global = total_abs_diff / total_t_items if total_t_items > 0 else float('inf')
    
    return wmape_global

def iterative_targeted_pairwise_swap(allocation_t_minus_1, allocation_t, factory_capacities, max_iterations=1500, max_time=600):
    start_time = time.time()
    target_recipe_counts = get_recipe_counts(allocation_t_minus_1, by_factory=True)
    current_allocation = {factory: orders[:] for factory, orders in allocation_t.items()}
    
    def evaluate_wmape(allocation):
        recipe_counts = get_recipe_counts(allocation, by_factory=True)
        total_abs_diff = sum(
            abs(recipe_counts[factory].get(recipe, 0) - target_recipe_counts[factory].get(recipe, 0))
            for factory in recipe_counts
            for recipe in set(recipe_counts[factory]) | set(target_recipe_counts[factory])
        )
        total_items = sum(sum(counts.values()) for counts in recipe_counts.values())
        return total_abs_diff / total_items if total_items > 0 else float('inf')
    
    def get_recipe_diff(allocation):
        recipe_counts = get_recipe_counts(allocation, by_factory=True)
        recipe_diff = defaultdict(lambda: defaultdict(int))
        for factory in recipe_counts:
            for recipe, count in recipe_counts[factory].items():
                recipe_diff[factory][recipe] = count - target_recipe_counts[factory].get(recipe, 0)
        return recipe_diff
    
    def find_swap_candidate(source_factory, target_factory, recipe_to_reduce, recipe_to_increase):
        source_orders = [order for order in current_allocation[source_factory] 
                         if recipe_to_reduce in order['recipe_ids'] 
                         and target_factory in order['eligible_factories']]
        target_orders = [order for order in current_allocation[target_factory] 
                         if recipe_to_increase in order['recipe_ids'] 
                         and source_factory in order['eligible_factories']]
        
        if source_orders and target_orders:
            return random.choice(source_orders), random.choice(target_orders)
        return None, None
    
    current_wmape = evaluate_wmape(current_allocation)
    best_allocation = current_allocation.copy()
    best_wmape = current_wmape

    for i in range(max_iterations):
        if time.time() - start_time > max_time:
            break
        
        recipe_diff = get_recipe_diff(current_allocation)
        factories = list(factory_capacities.keys())
        random.shuffle(factories)
        
        improved = False
        for source_factory in factories:
            if improved:
                break
            for target_factory in factories:
                if source_factory == target_factory:
                    continue
                
                recipes_to_reduce = [r for r, diff in recipe_diff[source_factory].items() if diff > 0]
                recipes_to_increase = [r for r, diff in recipe_diff[target_factory].items() if diff < 0]
                
                if recipes_to_reduce and recipes_to_increase:
                    recipe_to_reduce = random.choice(recipes_to_reduce)
                    recipe_to_increase = random.choice(recipes_to_increase)
                    
                    order1, order2 = find_swap_candidate(source_factory, target_factory, recipe_to_reduce, recipe_to_increase)
                    
                    if order1 and order2:
                        # Perform the swap
                        current_allocation[source_factory].remove(order1)
                        current_allocation[target_factory].remove(order2)
                        current_allocation[source_factory].append(order2)
                        current_allocation[target_factory].append(order1)
                        
                        new_wmape = evaluate_wmape(current_allocation)
                        
                        if new_wmape < current_wmape:
                            current_wmape = new_wmape
                            if new_wmape < best_wmape:
                                best_wmape = new_wmape
                                best_allocation = current_allocation.copy()
                            improved = True
                            break
                        else:
                            # Revert the swap
                            current_allocation[source_factory].remove(order2)
                            current_allocation[target_factory].remove(order1)
                            current_allocation[source_factory].append(order1)
                            current_allocation[target_factory].append(order2)

    return best_allocation, best_wmape

def run_allocation_process(quantity):
    global total_orders, factory_capacities
    
    # Set total orders and factory capacities
    total_orders = quantity
    F1_cap = int(0.25 * total_orders)
    F2_cap = int(0.5 * total_orders)
    factory_capacities = {
        'F1': F1_cap,
        'F2': F2_cap,
        'F3': float('inf')  # F3 has unlimited capacity
    }
    
    # Generate orders for day t-1
    orders_t_minus_1 = generate_orders_for_day(0.46, 0.54)
    # Generate orders for day t
    orders_t = generate_orders_for_day(0.52, 0.48, existing_real_orders=[o for o in orders_t_minus_1 if o['is_real']])
    # Perform allocation for day t-1 and day t
    allocation_t_minus_1 = allocate_orders(orders_t_minus_1, factory_capacities)
    allocation_t = allocate_orders(orders_t, factory_capacities)
    # Calculate WMAPE site before ITPS and WMAPE global
    wmape_site_before_itps = calculate_wmape_site(allocation_t_minus_1, allocation_t)
    wmape_global = calculate_wmape_global(allocation_t_minus_1, allocation_t)
    # Apply ITPS to optimize allocation on day t
    optimization_start_time = time.time()
    optimized_allocation_t, optimized_wmape_site = iterative_targeted_pairwise_swap(allocation_t_minus_1, allocation_t, factory_capacities)
    optimization_time = time.time() - optimization_start_time
        
    return optimization_time, wmape_site_before_itps, optimized_wmape_site, wmape_global

# Main execution code for scalability test
print("\nScalability test")
order_quantities = [10000, 30000, 50000, 70000, 100000]
results = []
for quantity in order_quantities:
    print(f"Running optimization for {quantity} orders...")
    optimization_time, wmape_before, wmape_after, wmape_global = run_allocation_process(quantity)
    results.append((quantity, optimization_time, wmape_before, wmape_after, wmape_global))
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print("--------------------")

# Print final results
print("Order quantity | Optimization time (s) | WMAPE site (Before ITPS) | WMAPE site (After ITPS) | WMAPE global")
print("---------------|------------------------|--------------------------|-------------------------|-------------")
for result in results:
    print(f"{result[0]:14d} | {result[1]:21.2f} | {result[2]:24.3f} | {result[3]:23.3f} | {result[4]:12.3f}")

# Plotting time
plt.figure(figsize=(12, 6))
plt.plot(order_quantities, [r[1] for r in results], marker='o')
plt.xlabel('Order quantity')
plt.ylabel('Optimization time (seconds)')
plt.title('Optimization time vs Order quantity')
plt.grid(True)
plt.show()

# Plotting WMAPE values
plt.figure(figsize=(12, 6))
plt.plot(order_quantities, [r[2] for r in results], marker='o', label='WMAPE site (Before ITPS)')
plt.plot(order_quantities, [r[3] for r in results], marker='s', label='WMAPE site (After ITPS)')
plt.plot(order_quantities, [r[4] for r in results], marker='^', label='WMAPE global')
plt.xlabel('Order quantity')
plt.ylabel('WMAPE')
plt.title('WMAPE vs Order quantity')
plt.legend()
plt.grid(True)
plt.show()