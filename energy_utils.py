import numpy as np
import pandas as pd
import os

# Priority Feeders Constants

TIER1 = {
    'SECURITY': ['sango', 'tollgate']          # Security critical
}
TIER2 = {
    'HEALTHCARE': ['idiroko']                    # Healthcare critical                        
}
TIER3 = {
    'FINANCIAL': ['fsm', 'qualitec', 'aarti', 'tower', 'arrachid', 'sumo']  # Financial services are Dedicated Lines
}
TIER4 = {
    'GENERAL': ['ijagba', 'amje', 'estate']        # General services; Residential, Industrial and Commercial
}

# Flattened tiers for easier processing
PRIORITY_ORDER = [
    *TIER1['SECURITY'],
    *TIER2['HEALTHCARE'],
    *TIER3['FINANCIAL'],
    *TIER4['GENERAL']
]

# Define transformer constraints
TRANSFORMERS = {
    'T1': {'capacity': 40, 'feeders': ['sango']},
    'T2': {'capacity': 60, 'feeders': ['fsm', 'amje', 'sumo']},
    'T3': {'capacity': 100, 'feeders': ['tower']},
    'T4': {'capacity': 60, 'feeders': ['qualitec', 'arrachid', 'ijagba', 'tollgate', 'aarti']},
    'T5': {'capacity': 40, 'feeders': ['idiroko', 'estate']}
}

# Feeder columns list
FEEDER_COLS = [
    'sango', 'idiroko', 'tollgate', 'fsm', 'ijagba', 'qualitec', 
    'sumo', 'aarti', 'tower', 'arrachid', 'amje', 'estate'
]

# Function to create cyclical features - this was highlighted in your selection
def create_cyclical_features(hour, day_of_week, month):
    """Create cyclical features for time variables"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    return hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos

# Function to prepare input features for model
def prepare_input_features(hour, day_of_week, month, day, year, is_weekend):
    """Prepare input features for the LSTM model"""
    # Create cyclical features
    hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos = create_cyclical_features(
        hour, day_of_week, month
    )
    
    # Create input array with all required features
    # Order matters and must match the training data
    input_features = np.array([
        hour, day_of_week, month, day, year, is_weekend,
        hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos
    ]).reshape(1, -1)
    
    return input_features

# Energy allocation function
def allocate_energy(forecasted_demand, available_supply):
    """Allocate available energy supply to feeders based on priority tiers"""
    total_demand = sum(forecasted_demand.values())
    allocation = {feeder: 0 for feeder in forecasted_demand}
    
    # Case 1: Supply exceeds or equals demand
    if available_supply >= total_demand:
        return forecasted_demand, "All feeders will receive 100% of their requested demand"
    
    # Case 2: Demand exceeds supply - allocate by priority
    allocation_log = []
    allocation_log.append(f"Demand ({total_demand:.2f} MW) exceeds supply ({available_supply:.2f} MW)")
    allocation_log.append("Allocating based on priority tiers...")
    
    remaining_supply = available_supply
    
    # Allocate by priority order
    for feeder in PRIORITY_ORDER:
        # Skip if this feeder is not in our forecast
        if feeder not in forecasted_demand:
            continue
            
        feeder_demand = forecasted_demand[feeder]
        
        # If we have enough supply for this feeder
        if remaining_supply >= feeder_demand:
            allocation[feeder] = feeder_demand
            remaining_supply -= feeder_demand
            allocation_log.append(f"  Allocated {feeder_demand:.2f} MW to {feeder} (100% of demand)")
        
        # If we have some supply but not enough for full demand
        elif remaining_supply > 0:
            allocation[feeder] = remaining_supply
            allocation_log.append(f"  Allocated {remaining_supply:.2f} MW to {feeder} ({remaining_supply/feeder_demand*100:.1f}% of demand)")
            remaining_supply = 0
        
        # If we're out of supply
        else:
            allocation_log.append(f"  No supply left for {feeder}")
    
    return allocation, "\n".join(allocation_log)

# Allocation analysis function
def analyze_allocation(allocation, forecasted_demand):
    """Analyze and report on the energy allocation"""
    total_allocated = sum(allocation.values())
    total_demand = sum(forecasted_demand.values())
    
    analysis = []
    analysis.append(f"Total Demand: {total_demand:.2f} MW")
    analysis.append(f"Total Allocated: {total_allocated:.2f} MW")
    analysis.append(f"Allocation Rate: {total_allocated/total_demand*100:.1f}%")
    
    tier_analysis = {}
    
    # Analyze by tier
    for tier_name, tier_dict in [("TIER1 (Security)", TIER1), 
                               ("TIER2 (Healthcare)", TIER2),
                               ("TIER3 (Financial)", TIER3), 
                               ("TIER4 (General)", TIER4)]:
        tier_feeders = [f for sublist in tier_dict.values() for f in sublist]
        tier_demand = sum(forecasted_demand.get(f, 0) for f in tier_feeders)
        tier_allocated = sum(allocation.get(f, 0) for f in tier_feeders)
        
        if tier_demand > 0:
            tier_analysis[tier_name] = {
                "demand": tier_demand,
                "allocated": tier_allocated,
                "satisfaction": tier_allocated/tier_demand*100
            }
            
            analysis.append(f"\n{tier_name}:")
            analysis.append(f"  Demand: {tier_demand:.2f} MW")
            analysis.append(f"  Allocated: {tier_allocated:.2f} MW")
            analysis.append(f"  Satisfaction Rate: {tier_allocated/tier_demand*100:.1f}%")
    
    # Check transformer constraints
    tx_analysis = {}
    analysis.append("\nTransformer Loading Analysis:")
    for tx_name, tx_info in TRANSFORMERS.items():
        tx_capacity = tx_info['capacity']
        tx_feeders = tx_info['feeders']
        tx_allocation = sum(allocation.get(f, 0) for f in tx_feeders)
        tx_utilization = tx_allocation / tx_capacity * 100
        
        tx_analysis[tx_name] = {
            "capacity": tx_capacity,
            "allocated": tx_allocation,
            "utilization": tx_utilization
        }
        
        analysis.append(f"Transformer {tx_name} ({tx_capacity} MW capacity):")
        analysis.append(f"  Feeders: {', '.join(tx_feeders)}")
        analysis.append(f"  Allocated: {tx_allocation:.2f} MW")
        analysis.append(f"  Utilization: {tx_utilization:.1f}%")
    
    return "\n".join(analysis), tier_analysis, tx_analysis

