import random
import time
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

def calculate_total_abs_diff(recipe_counts_t_minus_1, recipe_counts_t):
    all_recipes = set(recipe_counts_t_minus_1['F1'].keys()) | set(recipe_counts_t_minus_1['F2'].keys()) | set(recipe_counts_t_minus_1['F3'].keys()) | \
                  set(recipe_counts_t['F1'].keys()) | set(recipe_counts_t['F2'].keys()) | set(recipe_counts_t['F3'].keys())
    
    total_abs_diff = 0
    
    for recipe_id in all_recipes:
        for factory in ['F1', 'F2', 'F3']:
            t_minus_1_count = recipe_counts_t_minus_1[factory].get(recipe_id, 0)
            t_count = recipe_counts_t[factory].get(recipe_id, 0)
            total_abs_diff += abs(t_count - t_minus_1_count)
    
    return total_abs_diff

def tabu_search(allocation_t_minus_1, allocation_t, factory_capacities, max_iterations=100):
    current_allocation = {factory: orders[:] for factory, orders in allocation_t.items()}
    recipe_counts_t_minus_1 = get_recipe_counts(allocation_t_minus_1, by_factory=True)
    current_recipe_counts = get_recipe_counts(current_allocation, by_factory=True)
    current_total_abs_diff = calculate_total_abs_diff(recipe_counts_t_minus_1, current_recipe_counts)
    
    best_allocation = current_allocation.copy()
    best_total_abs_diff = current_total_abs_diff
    
    tabu_list = {}
    tabu_tenure = 20
    
    def swap_orders(factory1, order1, factory2, order2):
        current_allocation[factory1].remove(order1)
        current_allocation[factory2].remove(order2)
        current_allocation[factory1].append(order2)
        current_allocation[factory2].append(order1)
        
        for recipe_id in order1['recipe_ids']:
            current_recipe_counts[factory1][recipe_id] -= 1
            current_recipe_counts[factory2][recipe_id] = current_recipe_counts[factory2].get(recipe_id, 0) + 1
        for recipe_id in order2['recipe_ids']:
            current_recipe_counts[factory2][recipe_id] -= 1
            current_recipe_counts[factory1][recipe_id] = current_recipe_counts[factory1].get(recipe_id, 0) + 1
    
    def calculate_move_impact(order1, factory1, order2, factory2):
        diff = 0
        for recipe_id in order1['recipe_ids']:
            diff -= abs(current_recipe_counts[factory1][recipe_id] - recipe_counts_t_minus_1[factory1].get(recipe_id, 0))
            diff -= abs(current_recipe_counts[factory2].get(recipe_id, 0) - recipe_counts_t_minus_1[factory2].get(recipe_id, 0))
            diff += abs(current_recipe_counts[factory1][recipe_id] - 1 - recipe_counts_t_minus_1[factory1].get(recipe_id, 0))
            diff += abs(current_recipe_counts[factory2].get(recipe_id, 0) + 1 - recipe_counts_t_minus_1[factory2].get(recipe_id, 0))
        for recipe_id in order2['recipe_ids']:
            diff -= abs(current_recipe_counts[factory2][recipe_id] - recipe_counts_t_minus_1[factory2].get(recipe_id, 0))
            diff -= abs(current_recipe_counts[factory1].get(recipe_id, 0) - recipe_counts_t_minus_1[factory1].get(recipe_id, 0))
            diff += abs(current_recipe_counts[factory2][recipe_id] - 1 - recipe_counts_t_minus_1[factory2].get(recipe_id, 0))
            diff += abs(current_recipe_counts[factory1].get(recipe_id, 0) + 1 - recipe_counts_t_minus_1[factory1].get(recipe_id, 0))
        return diff
    
    for iteration in range(max_iterations):       
        best_move = None
        best_move_diff = 0
        
        # Randomly select a subset of orders to consider for swapping
        orders_to_consider = random.sample(sum(current_allocation.values(), []), min(100, len(sum(current_allocation.values(), []))))
        
        for order1 in orders_to_consider:
            factory1 = next(f for f, orders in current_allocation.items() if order1 in orders)
            for factory2 in factory_capacities:
                if factory1 != factory2:
                    for order2 in random.sample(current_allocation[factory2], min(10, len(current_allocation[factory2]))):
                        if factory2 in order1['eligible_factories'] and factory1 in order2['eligible_factories']:
                            move = (order1['id'], factory1, order2['id'], factory2)
                            if move not in tabu_list or current_total_abs_diff + calculate_move_impact(order1, factory1, order2, factory2) < best_total_abs_diff:
                                diff = calculate_move_impact(order1, factory1, order2, factory2)
                                if diff < best_move_diff:
                                    best_move = (order1, factory1, order2, factory2)
                                    best_move_diff = diff
        
        if best_move:
            order1, factory1, order2, factory2 = best_move
            swap_orders(factory1, order1, factory2, order2)
            current_total_abs_diff += best_move_diff
            
            if current_total_abs_diff < best_total_abs_diff:
                best_total_abs_diff = current_total_abs_diff
                best_allocation = {f: orders[:] for f, orders in current_allocation.items()}
            
            tabu_list[(order1['id'], factory1, order2['id'], factory2)] = iteration + tabu_tenure
            tabu_list[(order2['id'], factory2, order1['id'], factory1)] = iteration + tabu_tenure
        
        # Remove expired tabu moves
        tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
        
        # Diversification strategy
        if iteration % 100 == 0:
            factory1, factory2 = random.sample(list(factory_capacities.keys()), 2)
            order1 = random.choice(current_allocation[factory1])
            order2 = random.choice(current_allocation[factory2])
            if factory2 in order1['eligible_factories'] and factory1 in order2['eligible_factories']:
                swap_orders(factory1, order1, factory2, order2)
                current_total_abs_diff += calculate_move_impact(order1, factory1, order2, factory2)
    
    total_items_t = sum(sum(counts.values()) for counts in get_recipe_counts(best_allocation, by_factory=True).values())
    best_wmape_site = best_total_abs_diff / total_items_t if total_items_t > 0 else float('inf')

    return best_allocation, best_wmape_site

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

def test_ts_iterations(allocation_t_minus_1, allocation_t, factory_capacities, max_iterations_list):
    results = []
    
    for max_iterations in max_iterations_list:
        start_time = time.time()
        optimized_allocation, optimized_wmape_site = tabu_search(
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(iterations, wmape_site_values, marker='o', label='WMAPE site')
    ax1.axhline(y=wmape_global, color='r', linestyle='--', label='WMAPE global')
    ax1.set_ylabel('WMAPE')
    ax1.set_title('WMAPE site vs. Number of iterations (TS)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(iterations, times, marker='s')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Optimization time (seconds)')
    ax2.set_title('Optimization time vs. Number of iterations (TS)')
    ax2.grid(True)
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
iteration_counts = [100, 500, 1000, 1500, 2000]

results = test_ts_iterations(
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