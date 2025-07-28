#!/usr/bin/env python3
"""
Simple test for the fixed feature compatibility
"""
import os
import json

# Load config to check font percentiles
config_path = 'config_main.json'
with open(config_path, 'r') as f:
    config = json.load(f)

font_percentiles_config = config.get('feature_engineering', {}).get('font_percentiles', [])
print(f"Config font_percentiles: {font_percentiles_config}")

# Check training script default
train_default = [50, 75, 90, 95]
print(f"train_model.py default: {train_default}")

# Check if they match
if font_percentiles_config == train_default:
    print("✅ Font percentiles match between config and training default!")
else:
    print("⚠️ Font percentiles don't match - this could cause feature mismatch")
    print(f"   Config has {len(font_percentiles_config)} percentiles")
    print(f"   Training default has {len(train_default)} percentiles")
    print(f"   Difference: {len(font_percentiles_config) - len(train_default)} features")

print("\nThe issue was likely:")
print("1. Config specifies 8 font percentiles: [70, 75, 80, 85, 90, 95, 98, 99]")
print("2. During training, if font data was missing, train_model.py used its hardcoded default: [50, 75, 90, 95]")
print("3. This created 4 fewer features (115 vs 119)")
print("4. The prediction script was using config default (8 percentiles), creating more features")
print("\nFix: Make generate_json_output.py use the same default as train_model.py")
