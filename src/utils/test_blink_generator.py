# src/utils/test_blink_generator.py
import random
print("--- 🛰️ Task 18: Occlusion Generator Test ---")

# Simulate 20 frames of data generation
for frame in range(1, 21):
    # Simulate a 10% blink rate for testing
    is_occluded = random.random() < 0.10 
    status = "🕶️ HIDDEN (Blink)" if is_occluded else "👁️ VISIBLE"
    
    print(f"Frame {frame:02d}: {status}")

print("\n✅ Task 18 Logic Ready. Ready to generate Hardened Dataset.")