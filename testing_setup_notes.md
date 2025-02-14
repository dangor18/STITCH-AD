# Hyperparam tuning for:
-  UNET base model if used
-  SAM model for both preprocessed and unprocessed methods
- SAM testing with Autumn filtered Images

# Some kind of k-fold validation using orchards as sections of the dataset
Need some kind of Train/Test/Validation with:
- either mutually exclusive sets (different orchards)
- or mutually exclusive sets of chunks (same orchards, different parts)
- Respect percentages of different classes in each set (train, val, test)
- 

# Missing Test Masks:
- 2222
- 2227
- 2228
- 2240
- 2848

# Option 1: Orchard based k-fold validation split
## Total orchards: 9 (10 with case 3 orchard)
Deal with non-representative sets of data??
*Folds: 5*
**Apply Folds Using Metadata Generation Files**
Train/Val Orchards: 1724, 1883, 2222, 2227, 2240, 2848
Train: 4 (or 5?)
Validation: 2 (or 1?) - Options: 1676, 1996, 2057, 1883

Test: 3 - Options: 1676, 1996, 2057, 1883

Orchard types:
Dots: 1676, 
Lines: 1724, 1996, 2057, 2222, 2228, 2240, 2848 
Covered lines: 2227,

Defective orchards: 1676, 1724, (1883), 1996, 2057 - ???



Monte Carlo Cross Validation:
- Divide at random into 5 sets
- Train on 4, Validate on 1
- Mutually exclusive sets