import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# Define the recipes with their eligibility
recipes_f1_only = list(range(1, 30))
recipes_f1_f2 = list(range(30, 50))
recipes_f2_only = list(range(50, 90))
recipes_f3_only = list(range(90, 101))

# Define total orders and factory capacities
total_orders = 10000
F1_cap = int(0.25 * total_orders)  # 25% of total orders
F2_cap = int(0.5 * total_orders)  # 50% of total orders
factory_capacities = {
    'F1': F1_cap,
    'F2': F2_cap,
    'F3': float('inf')  # F3 has unlimited capacity
}

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

def iterative_targeted_pairwise_swap(allocation_t_minus_1, allocation_t, factory_capacities, max_iterations=500):   
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

def calculate_wmape_site(allocation_t_minus_1, allocation_t):
    recipe_counts_t_minus_1 = get_recipe_counts(allocation_t_minus_1, by_factory=True)
    recipe_counts_t = get_recipe_counts(allocation_t, by_factory=True)
    
    all_recipes = set(recipe_counts_t_minus_1['F1'].keys()) | set(recipe_counts_t_minus_1['F2'].keys()) | set(recipe_counts_t_minus_1['F3'].keys()) | \
                  set(recipe_counts_t['F1'].keys()) | set(recipe_counts_t['F2'].keys()) | set(recipe_counts_t['F3'].keys())
    
    total_abs_diff = 0
    total_items_t = 0
    
    for recipe_id in all_recipes:
        for factory in ['F1', 'F2', 'F3']:
            t_minus_1_count = recipe_counts_t_minus_1[factory].get(recipe_id, 0)
            t_count = recipe_counts_t[factory].get(recipe_id, 0)
            total_abs_diff += abs(t_count - t_minus_1_count)
            total_items_t += t_count
    
    wmape_site = total_abs_diff / total_items_t if total_items_t > 0 else float('inf')
    
    return wmape_site

def calculate_wmape_global(allocation_t_minus_1, allocation_t):
    recipe_counts_t_minus_1 = get_recipe_counts(allocation_t_minus_1)
    recipe_counts_t = get_recipe_counts(allocation_t)
    
    all_recipes = set(recipe_counts_t_minus_1.keys()) | set(recipe_counts_t.keys())
    
    total_abs_diff = 0
    total_t_items = sum(recipe_counts_t.values())
    
    for recipe_id in all_recipes:
        t_minus_1_count = recipe_counts_t_minus_1.get(recipe_id, 0)
        t_count = recipe_counts_t.get(recipe_id, 0)
        total_abs_diff += abs(t_count - t_minus_1_count)
    
    wmape_global = total_abs_diff / total_t_items if total_t_items > 0 else float('inf')
    
    return wmape_global

def test_itps_iterations(allocation_t_minus_1, allocation_t, factory_capacities, max_iterations_list):
    results = []
    
    for max_iterations in max_iterations_list:
        start_time = time.time()
        optimized_allocation, optimized_wmape_site = iterative_targeted_pairwise_swap(
            allocation_t_minus_1, 
            allocation_t, 
            factory_capacities, 
            max_iterations=max_iterations
        )
        end_time = time.time()
        optimization_time = end_time - start_time
        results.append((max_iterations, optimized_wmape_site, optimization_time))
    
    return results

def plot_iteration_results(results, initial_wmape_site, wmape_global):
    iterations, wmape_site_values, times = zip(*results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(iterations, wmape_site_values, marker='o', label='WMAPE site')
    ax1.axhline(y=wmape_global, color='r', linestyle='--', label='WMAPE global')
    ax1.set_ylabel('WMAPE')
    ax1.set_title('WMAPE site vs. Number of iterations (ITPS)')
    ax1.legend()
    ax1.grid(True)
    
    # Adjust y-axis limits to focus more tightly on the data range
    y_min = min(min(wmape_site_values), wmape_global) * 0.999  # 0.1% below the minimum value
    y_max = max(max(wmape_site_values), wmape_global) * 1.001  # 0.1% above the maximum value
    ax1.set_ylim(y_min, y_max)
    
    # Set x-axis ticks and labels for WMAPE plot
    ax1.set_xticks(iterations)
    ax1.set_xticklabels(iterations)
    
    ax2.plot(iterations, times, marker='s')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Optimization time (seconds)')
    ax2.set_title('Optimization time vs. Number of iterations (ITPS)')
    ax2.grid(True)
    ax2.set_ylim(bottom=0)
    
    # Set x-axis limits and ticks for both plots
    min_iterations = min(iterations)
    max_iterations = max(iterations)
    x_padding = (max_iterations - min_iterations) * 0.05  # 5% padding
    ax1.set_xlim(min_iterations - x_padding, max_iterations + x_padding)
    ax2.set_xlim(min_iterations - x_padding, max_iterations + x_padding)
    ax2.set_xticks(iterations)
    
    plt.tight_layout()
    plt.show()

# Main execution code (day t-1 is LD12, day t is LD11)
orders_day_t_minus_1 = generate_orders_for_day(0.46, 0.54)
real_orders_day_t_minus_1 = [order for order in orders_day_t_minus_1 if order['is_real']]

orders_day_t = generate_orders_for_day(0.52, 0.48, existing_real_orders=real_orders_day_t_minus_1)

allocation_day_t_minus_1 = allocate_orders(orders_day_t_minus_1, factory_capacities)
allocation_day_t = allocate_orders(orders_day_t, factory_capacities)

# Calculate initial WMAPE site and WMAPE global
initial_wmape_site = calculate_wmape_site(allocation_day_t_minus_1, allocation_day_t)
wmape_global = calculate_wmape_global(allocation_day_t_minus_1, allocation_day_t)

# Define the list of iteration counts to test
iteration_counts = [500, 1000, 1500, 2000, 2500]

results = test_itps_iterations(
    allocation_day_t_minus_1,
    allocation_day_t,
    factory_capacities,
    iteration_counts
)

plot_iteration_results(results, initial_wmape_site, wmape_global)

# Find the best result
best_result = min(results, key=lambda x: x[1])
best_iterations, best_wmape_site, best_time = best_result

print(f"\nInitial WMAPE site: {initial_wmape_site:.4f}")
print(f"WMAPE global: {wmape_global:.4f}")
print(f"Best WMAPE site: {best_wmape_site:.4f} (achieved with {best_iterations} iterations)")
print(f"WMAPE site improvement: {(initial_wmape_site - best_wmape_site) / initial_wmape_site * 100:.2f}%")
print(f"Time taken for best result: {best_time:.2f} seconds")
print('                               ')

# Print all results
print("{:<10} | {:<18} | {:<20} | {:<26} | {:<12} | {:<15}".format(
    "Iterations", "Initial WMAPE site", "Optimized WMAPE site", "WMAPE site improvement (%)", "WMAPE global", "Time (seconds)"
))
print("-" * 110)
for iterations, wmape_site, time_taken in results:
    wmape_site_improvement = (initial_wmape_site - wmape_site) / initial_wmape_site * 100
    print("{:<10d} | {:<18.4f} | {:<20.4f} | {:<26.2f} | {:<12.4f} | {:<15.2f}".format(
        iterations, initial_wmape_site, wmape_site, wmape_site_improvement, wmape_global, time_taken
    ))