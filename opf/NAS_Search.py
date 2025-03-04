import random

# Define the search space for NAS-based search


def generate_random_genotype():
    """Generates a random network architecture for NAS-based search."""
    operations = ['LSTM', 'GRU', 'CNN', 'Transformer', 'Identity']
    
    genotype = []
    for _ in range(random.randint(3, 6)):  # Randomly select 3 to 6 layers
        op_name = random.choice(operations)
        if op_name in ['LSTM', 'GRU']:
            genotype.append((op_name, {'num_layers': random.randint(1, 3)}))
        elif op_name == 'CNN':
            genotype.append((op_name, {'kernel_size': random.choice([3, 5, 7])}))
        elif op_name == 'Transformer':
            genotype.append((op_name, {'nhead': random.choice([2, 4, 8])}))
        elif op_name == 'Identity':
            genotype.append((op_name, {}))

    return genotype
