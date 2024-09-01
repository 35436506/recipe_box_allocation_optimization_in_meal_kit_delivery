from pulp import *
import random
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook

# Set random seed for reproducibility
random.seed(42)

# Define the recipes with their eligibility
recipes_f1_only = list(range(1, 30))
recipes_f1_f2 = list(range(30, 50))
recipes_f2_only = list(range(50, 90))
recipes_f3_only = list(range(90, 101))

# Define total orders and initial factory capacities
total_orders = 10000
initial_F1_cap = 3000
F2_cap = int(0.5 * total_orders)  # 50% of total orders

def get_factory_capacities(day):
    if day >= -6:  # LD7 and onwards
        return {
            'F1': 1000,
            'F2': 6000,
            'F3': float('inf')
        }
    elif day >= -10:  # LD10 to LD8
        return {
            'F1': 1000,
            'F2': F2_cap,
            'F3': float('inf')
        }
    else:  # Before LD10
        return {
            'F1': initial_F1_cap,
            'F2': F2_cap,
            'F3': float('inf')
        }

def generate_order_recipes(eligible_recipes, max_recipes=4):
    return random.sample(eligible_recipes, random.randint(1, min(max_recipes, len(eligible_recipes))))

def generate_orders_for_day(real_proportion, simulated_proportion, existing_real_orders=None):
    orders = []
    f1_f3_target = int(0.3 * total_orders)
    f2_f3_target = int(0.6 * total_orders)
    f1_f2_f3_target = int(0.1 * total_orders)
    
    total_real_orders = int(real_proportion * total_orders)
    
    changed_orders = []
    deleted_orders = []
    
    # Create a set of all available IDs
    available_ids = set(range(1, total_orders + 1))
    
    if existing_real_orders:
        # Delete 5% of real orders
        delete_count = int(0.05 * len(existing_real_orders))
        deleted_orders = random.sample(existing_real_orders, delete_count)
        remaining_orders = [order for order in existing_real_orders if order not in deleted_orders]
        
        # Change 30% of remaining real orders
        change_count = int(0.3 * len(remaining_orders))
        changed_orders = random.sample(remaining_orders, change_count)
        
        for order in remaining_orders:
            if order in changed_orders:
                # Change recipe composition and eligibility
                new_recipes = generate_order_recipes(recipes_f1_only + recipes_f1_f2 + recipes_f2_only + recipes_f3_only)
                new_eligibility = ['F3']  # Always eligible for F3
                if any(r in recipes_f1_only + recipes_f1_f2 for r in new_recipes):
                    new_eligibility.append('F1')
                if any(r in recipes_f1_f2 + recipes_f2_only for r in new_recipes):
                    new_eligibility.append('F2')
                order['recipe_ids'] = new_recipes
                order['eligible_factories'] = new_eligibility
            orders.append(order)
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
    
    return orders, changed_orders, deleted_orders

def BB_allocation(orders, factory_capacities, recipe_counts_t_minus_1):
    prob = LpProblem("Order_Allocation", LpMinimize)
    
    x = LpVariable.dicts("allocation", 
                         [(o['id'], f) for o in orders for f in factory_capacities.keys()], 
                         cat='Binary')
    
    for o in orders:
        prob += lpSum(x[o['id'], f] for f in factory_capacities.keys()) == 1
    
    for o in orders:
        for f in factory_capacities.keys():
            if f not in o['eligible_factories']:
                prob += x[o['id'], f] == 0
    
    prob += lpSum(x[o['id'], 'F1'] for o in orders) == factory_capacities['F1']
    prob += lpSum(x[o['id'], 'F2'] for o in orders) == factory_capacities['F2']
    
    recipe_counts_t = {}
    for f in factory_capacities.keys():
        for o in orders:
            for r in o['recipe_ids']:
                if r not in recipe_counts_t:
                    recipe_counts_t[r] = {}
                if f not in recipe_counts_t[r]:
                    recipe_counts_t[r][f] = 0
                recipe_counts_t[r][f] += x[o['id'], f]
    
    abs_diffs = []
    for r in set(recipe_counts_t.keys()) | set(sum((list(counts.keys()) for counts in recipe_counts_t_minus_1.values()), [])):
        for f in factory_capacities.keys():
            t_minus_1_count = recipe_counts_t_minus_1[f].get(r, 0)
            t_count = recipe_counts_t.get(r, {}).get(f, 0)
            
            pos_diff = LpVariable(f"pos_diff_{r}_{f}", lowBound=0)
            neg_diff = LpVariable(f"neg_diff_{r}_{f}", lowBound=0)
            
            prob += t_count - t_minus_1_count == pos_diff - neg_diff
            
            abs_diffs.append(pos_diff + neg_diff)
    
    prob += lpSum(abs_diffs)
    
    prob.solve()
    
    allocation = {f: [] for f in factory_capacities.keys()}
    for o in orders:
        for f in factory_capacities.keys():
            if value(x[o['id'], f]) == 1:
                allocation[f].append(o['id'])
    
    return allocation

def allocate_orders(orders, factory_capacities):
    allocation = {factory: [] for factory in factory_capacities}
    remaining_orders = orders.copy()
    # Allocate to F1 first, prioritizing F1,F3 and F1,F2,F3 orders
    f1_eligible = sorted([order for order in remaining_orders if 'F1' in order['eligible_factories']],
                         key=lambda x: len(x['eligible_factories']))  # Prioritize F1,F3 over F1,F2,F3
    allocation['F1'] = [order['id'] for order in f1_eligible[:factory_capacities['F1']]]
    remaining_orders = [order for order in remaining_orders if order['id'] not in allocation['F1']]
    # Allocate to F2, using remaining F1,F2,F3 orders and F2,F3 orders
    f2_eligible = [order for order in remaining_orders if 'F2' in order['eligible_factories']]
    allocation['F2'] = [order['id'] for order in f2_eligible[:factory_capacities['F2']]]
    remaining_orders = [order for order in remaining_orders if order['id'] not in allocation['F2']]
    # Allocate remaining to F3
    allocation['F3'] = [order['id'] for order in remaining_orders]
    return allocation

def get_recipe_counts(orders, allocation):
    recipe_counts = {f: {} for f in allocation.keys()}
    for factory, order_ids in allocation.items():
        for order_id in order_ids:
            order = next((o for o in orders if o['id'] == order_id), None)
            
            if order is None:
                continue
            
            for recipe in order['recipe_ids']:
                if recipe not in recipe_counts[factory]:
                    recipe_counts[factory][recipe] = 0
                recipe_counts[factory][recipe] += 1
    return recipe_counts

def calculate_wmape_site(orders_t_minus_1, allocation_t_minus_1, orders_t, allocation_t):
    total_abs_diff = 0
    total_items_t = 0
    
    recipe_counts_t_minus_1 = get_recipe_counts(orders_t_minus_1, allocation_t_minus_1)
    recipe_counts_t = get_recipe_counts(orders_t, allocation_t)
    
    for factory in ['F1', 'F2', 'F3']:
        for recipe_id in set(recipe_counts_t_minus_1[factory].keys()) | set(recipe_counts_t[factory].keys()):
            t_minus_1_count = recipe_counts_t_minus_1[factory].get(recipe_id, 0)
            t_count = recipe_counts_t[factory].get(recipe_id, 0)
            abs_diff = abs(t_count - t_minus_1_count)
            total_abs_diff += abs_diff
            total_items_t += t_count
    
    if total_items_t == 0:
        wmape_site = float('inf')
    else:
        wmape_site = total_abs_diff / total_items_t
    
    return wmape_site

def calculate_wmape_global(orders_t_minus_1, allocation_t_minus_1, orders_t, allocation_t):
    total_abs_diff = 0
    total_t_items = 0
    recipe_counts_t_minus_1 = {}
    recipe_counts_t = {}
    
    for factory in allocation_t_minus_1:
        for order_id in allocation_t_minus_1[factory]:
            order = next((o for o in orders_t_minus_1 if o['id'] == order_id), None)
            if order:
                for recipe_id in order['recipe_ids']:
                    recipe_counts_t_minus_1[recipe_id] = recipe_counts_t_minus_1.get(recipe_id, 0) + 1
    
    for factory in allocation_t:
        for order_id in allocation_t[factory]:
            order = next((o for o in orders_t if o['id'] == order_id), None)
            if order:
                for recipe_id in order['recipe_ids']:
                    recipe_counts_t[recipe_id] = recipe_counts_t.get(recipe_id, 0) + 1
                    total_t_items += 1
    
    all_recipes = set(recipe_counts_t_minus_1.keys()) | set(recipe_counts_t.keys())
    
    for recipe_id in all_recipes:
        t_minus_1_count = recipe_counts_t_minus_1.get(recipe_id, 0)
        t_count = recipe_counts_t.get(recipe_id, 0)
        abs_diff = abs(t_minus_1_count - t_count)
        total_abs_diff += abs_diff
    
    if total_t_items == 0:
        wmape_global = float('inf')
    else:
        wmape_global = total_abs_diff / total_t_items
    
    return wmape_global

def calculate_wmape_site_between_allocations(orders_1, allocation_1, orders_2, allocation_2):
    recipe_counts_1 = get_recipe_counts(orders_1, allocation_1)
    recipe_counts_2 = get_recipe_counts(orders_2, allocation_2)
    
    total_abs_diff = 0
    total_items_2 = 0
    
    for factory in ['F1', 'F2', 'F3']:
        for recipe_id in set(recipe_counts_1[factory].keys()) | set(recipe_counts_2[factory].keys()):
            count_1 = recipe_counts_1[factory].get(recipe_id, 0)
            count_2 = recipe_counts_2[factory].get(recipe_id, 0)
            abs_diff = abs(count_2 - count_1)
            total_abs_diff += abs_diff
            total_items_2 += count_2
    
    if total_items_2 == 0:
        return float('inf')
    else:
        return total_abs_diff / total_items_2

def run_allocation_process_over_time(start_day, end_day, total_orders):
    allocations_bb = {}
    allocations_greedy = {}
    wmape_site_values_bb = []
    wmape_site_values_greedy = []
    wmape_global_values = []
    real_orders_proportions = []
    previous_real_orders = None

    for day in range(start_day, end_day + 1):
        real_proportion = min(1.0, max(0.1, 0.1 + (0.9 / 15) * (18 + day)))
        simulated_proportion = 1 - real_proportion

        orders, changed_orders, deleted_orders = generate_orders_for_day(real_proportion, simulated_proportion, previous_real_orders)
        factory_capacities = get_factory_capacities(day)

        if day > start_day:
            recipe_counts_t_minus_1 = get_recipe_counts(allocations_bb[day-1]['orders'], allocations_bb[day-1]['allocation'])
            allocation_bb = BB_allocation(orders, factory_capacities, recipe_counts_t_minus_1)
            allocation_greedy = allocate_orders(orders, factory_capacities)

            wmape_site_bb = calculate_wmape_site(
                allocations_bb[day-1]['orders'],
                allocations_bb[day-1]['allocation'],
                orders,
                allocation_bb
            )
            wmape_site_greedy = calculate_wmape_site(
                allocations_greedy[day-1]['orders'],
                allocations_greedy[day-1]['allocation'],
                orders,
                allocation_greedy
            )

            wmape_global = calculate_wmape_global(
                allocations_bb[day-1]['orders'],
                allocations_bb[day-1]['allocation'],
                orders,
                allocation_bb
            )
            wmape_site_values_bb.append(wmape_site_bb)
            wmape_site_values_greedy.append(wmape_site_greedy)
            wmape_global_values.append(wmape_global)
        else:
            allocation_bb = allocate_orders(orders, factory_capacities)
            allocation_greedy = allocate_orders(orders, factory_capacities)

        allocations_bb[day] = {'orders': orders, 'allocation': allocation_bb}
        allocations_greedy[day] = {'orders': orders, 'allocation': allocation_greedy}
        real_orders_proportions.append(len([o for o in orders if o['is_real']]) / total_orders)
        previous_real_orders = [order for order in orders if order['is_real']]

    return allocations_bb, allocations_greedy, wmape_site_values_bb, wmape_site_values_greedy, wmape_global_values, real_orders_proportions

def plot_temporal_component(days, real_orders_proportions):
    plt.figure(figsize=(15, 8))
    
    real_orders = []
    simulated_orders = []
    
    for prop in real_orders_proportions:
        total_real = int(prop * total_orders)
        real_orders.append(total_real)
        simulated_orders.append(total_orders - total_real)
    
    x = np.arange(len(days))
    width = 0.7
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x, real_orders, width, label='Real orders', color='blue')
    ax.bar(x, simulated_orders, width, bottom=real_orders, label='Simulated orders', color='orange')
    
    ax.set_xlabel('Days to delivery', fontsize=12)
    ax.set_ylabel('Order quantity', fontsize=12)
    ax.set_title('Composition of total orders through days', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(day) for day in days], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    ax.set_ylim(0, 1.1 * total_orders)
    
    ax.axvline(x=16, color='purple', linestyle='--', linewidth=1)
    ax.text(16, ax.get_ylim()[1], 'Delivery date', va='bottom', ha='left', color='purple')
    ax.axvline(x=-1, color='purple', linestyle=':', linewidth=1)
    ax.text(-1, ax.get_ylim()[1], 'Menu opens', va='bottom', ha='left', color='purple')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_wmape_over_time(days, wmape_site_values_bb, wmape_site_values_greedy, wmape_global_values):
    plt.figure(figsize=(12, 6))
    plt.plot(days[1:], wmape_site_values_greedy, label='WMAPE site (Greedy)', marker='s')
    plt.plot(days[1:], wmape_site_values_bb, label='WMAPE site (B&B)', marker='o')
    plt.plot(days[1:], wmape_global_values, label='WMAPE global', marker='^')
    
    plt.xlabel('Days to delivery')
    plt.ylabel('WMAPE')
    plt.title("WMAPE site and global through days", pad=20, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.axvline(x=-10, color='red', linestyle='--', linewidth=1)
    plt.text(-10, plt.ylim()[1], 'F1 capacity decreases', ha='right', va='bottom', color='red', rotation=0, fontsize=10)
    plt.axvline(x=-6, color='purple', linestyle='--', linewidth=1)
    plt.text(-6, plt.ylim()[1], 'F2 capacity increases', ha='right', va='bottom', color='purple', rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_factory_allocations(days, allocations_bb):
    plt.figure(figsize=(15, 8))
    
    f1_allocations = []
    f2_allocations = []
    f3_allocations = []
    
    for day in days:
        allocation = allocations_bb[day]['allocation']
        f1_allocations.append(len(allocation['F1']))
        f2_allocations.append(len(allocation['F2']))
        f3_allocations.append(len(allocation['F3']))
    
    plt.plot(days, f1_allocations, label='F1', color='blue', marker='o')
    plt.plot(days, f2_allocations, label='F2', color='orange', marker='s')
    plt.plot(days, f3_allocations, label='F3', color='green', marker='^')
    
    plt.xlabel('Days to delivery', fontsize=12)
    plt.ylabel('Total number of orders', fontsize=12)
    plt.title("Total orders allocated to each factory through days", pad=20, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.axvline(x=-10, color='red', linestyle='--', linewidth=1)
    plt.text(-10, plt.ylim()[1], 'F1 capacity decreases', ha='right', va='bottom', color='red', rotation=0, fontsize=10)
    
    plt.axvline(x=-6, color='purple', linestyle='--', linewidth=1)
    plt.text(-6, plt.ylim()[1], 'F2 capacity increases', ha='right', va='bottom', color='purple', rotation=0, fontsize=10)
    
    plt.xlim(min(days), max(days))
    plt.xticks(range(min(days), max(days)+1, 2))
    
    plt.tight_layout()
    plt.show()

def plot_wmape_vs_final(days, wmape_site_vs_final_bb, wmape_global_vs_final):
    plt.figure(figsize=(12, 6))
    plt.plot(days, wmape_site_vs_final_bb, label='WMAPE site', marker='o')
    plt.plot(days, wmape_global_vs_final, label='WMAPE global', marker='s')
    plt.xlabel('Days to delivery')
    plt.ylabel('WMAPE')
    plt.title("Comparing each day's allocation with LD3's allocation", pad=20, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.axvline(x=-10, color='red', linestyle='--', linewidth=1)
    plt.text(-10, plt.ylim()[1], 'F1 capacity decreases', ha='right', va='bottom', color='red', rotation=0, fontsize=10)
    plt.axvline(x=-6, color='purple', linestyle='--', linewidth=1)
    plt.text(-6, plt.ylim()[1], 'F2 capacity increases', ha='right', va='bottom', color='purple', rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()
    
def calculate_wmape_global_vs_final(allocations, final_day):
    wmape_global_vs_final = []
    for day in allocations.keys():
        wmape = calculate_wmape_global(
            allocations[day]['orders'],
            allocations[day]['allocation'],
            allocations[final_day]['orders'],
            allocations[final_day]['allocation'])
        wmape_global_vs_final.append(wmape)
    return wmape_global_vs_final

def export_to_excel(days, real_orders_proportions, wmape_site_values_bb, wmape_site_values_greedy, wmape_global_values, wmape_site_vs_final_bb, wmape_global_vs_final):
    wb = Workbook()
    ws = wb.active
    ws.title = "WMAPE results"

    headers = ["Day", "Real orders proportion", "WMAPE site (B&B)", "WMAPE site (Greedy)", "WMAPE global", "WMAPE site vs LD3", "WMAPE global vs LD3"]
    ws.append(headers)

    for i, day in enumerate(days):
        row = [
            day,
            real_orders_proportions[i],
            wmape_site_values_bb[i-1] if i > 0 else "N/A",
            wmape_site_values_greedy[i-1] if i > 0 else "N/A",
            wmape_global_values[i-1] if i > 0 else "N/A",
            wmape_site_vs_final_bb[i],
            wmape_global_vs_final[i]
        ]
        ws.append(row)

    wb.save("Both capacity and order change.xlsx")

# Main execution code
start_day = -18
end_day = -3

allocations_bb, allocations_greedy, wmape_site_values_bb, wmape_site_values_greedy, wmape_global_values, real_orders_proportions = run_allocation_process_over_time(start_day, end_day, total_orders)

days = list(range(start_day, end_day + 1))

# Plot 1: Temporal component of order allocation
plot_temporal_component(days, real_orders_proportions)

# Plot 2: Factory allocations over time
plot_factory_allocations(days, allocations_bb)

# Plot 3: WMAPE over time
plot_wmape_over_time(days, wmape_site_values_bb, wmape_site_values_greedy, wmape_global_values)

# Calculate WMAPE site vs final allocation for B&B
final_day = end_day
wmape_site_vs_final_bb = []
for day in range(start_day, end_day + 1):
    wmape_bb = calculate_wmape_site_between_allocations(
        allocations_bb[day]['orders'],
        allocations_bb[day]['allocation'],
        allocations_bb[final_day]['orders'],
        allocations_bb[final_day]['allocation'])
    wmape_site_vs_final_bb.append(wmape_bb)

# Calculate WMAPE global vs final allocation
wmape_global_vs_final = calculate_wmape_global_vs_final(allocations_bb, final_day)

# Plot 4: WMAPE site and global comparison
plot_wmape_vs_final(days, wmape_site_vs_final_bb, wmape_global_vs_final)

# Export results to Excel
export_to_excel(days, real_orders_proportions, wmape_site_values_bb, wmape_site_values_greedy, wmape_global_values, wmape_site_vs_final_bb, wmape_global_vs_final)