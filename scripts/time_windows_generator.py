import numpy as np

def sample_time_window(customer_type, customer_appearance_time):
    """
    Samples a realistic time window for delivering products to a customer.
    
    Parameters:
    - customer_type: 0 ('residential') or 1 ('commercial'), indicating the type of customer.
    - customer_appearance_time: appearance time of a dynamic customer on the map.
    
    Returns:
    - A tuple (start_time, end_time) representing the delivery time window in integer hours.
    """
    
    # Peak times in minutes from midnight
    morning_peak = 8 * 60    # 8:00 AM
    evening_peak = 19 * 60   # 7:00 PM
    business_peak = 13 * 60  # 1:00 PM for commercial
    
    # Delivery day constraints
    delivery_day_start = 7 * 60   # 7:00 AM
    delivery_day_end = 24 * 60    # 9:00 PM (21:00)
    # Delivery day can't be earlier than dynamic customer appearance time 
    delivery_day_start = max(delivery_day_start, customer_appearance_time)
    
    if customer_type == 0:
        # Residential customers: Normal distribution around morning and evening peaks
        if np.random.uniform(0, 1) < 0.5:
            # Morning window: sample starting time from a normal distribution around 8 AM
            start_time = np.random.normal(loc=morning_peak, scale=90)  # Scale 60 = 1 hour
        else:
            # Evening window: sample starting time from a normal distribution around 7 PM
            start_time = np.random.normal(loc=evening_peak, scale=120)
        
        # Ensure start time is within delivery hours
        start_time = max(delivery_day_start, min(start_time, delivery_day_end - 60))
        
        # Residential customers usually accept longer windows, sample integer length
        window_length = int(np.random.uniform(1, 3)) * 60  # Length in integer hours (1 to 3 hours)
    
    elif customer_type == 1:
        # Commercial customers: Normal distribution around 1 PM
        start_time = np.random.normal(loc=business_peak, scale=90)
        
        # Ensure start time is within delivery hours
        start_time = max(delivery_day_start, min(start_time, delivery_day_end - 60))
        
        # Commercial customers prefer shorter windows, sample integer length
        window_length = int(np.random.uniform(1, 2)) * 60  # Length in integer hours (1 to 2 hours)
    
    else:
        raise ValueError("customer_type must be 0 ('residential') or 1 ('commercial')")
    
    # Calculate end time based on the start time and window length
    end_time = start_time + window_length
    
    # Ensure the end time doesn't exceed the delivery day end
    end_time = min(end_time, delivery_day_end)
    
    return (start_time, end_time)

# Example usage:
if __name__ == "__main__":
    num_customers = 5
    for _ in range(num_customers):
        customer_type = np.random.choice([0, 1])
        customer_appearance_time = np.random.uniform(0, 18*60)
        time_window = sample_time_window(customer_type, customer_appearance_time)
        start_time, end_time = time_window
        # Convert times to hours for readability
        start_time_hours = int(start_time) // 60
        end_time_hours = int(end_time) // 60
        print(f"Customer Type: {customer_type}, Delivery Window: {start_time_hours:02d}:00 - {end_time_hours:02d}:00")
