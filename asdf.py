X = {1, 2, 3}
Y = {2, 3, 4}

# Check the identity of X before the intersection operation
original_id = id(X)

# Perform intersection operation and check identity of X again
X &= Y
new_id = id(X)

# Compare the identity of X before and after intersection operation
if original_id == new_id:
    print("X has not been updated")
else:
    print("X has been updated")