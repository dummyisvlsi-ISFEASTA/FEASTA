
import sys
import os

# Add pysta to path (relative to project root)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from pysta import Design

def main():
    print("\n" + "="*60)
    print("PySTA Verification: ZipCPU Design")
    print("="*60)

    # 1. Load Design
    print("\n[1] Loading Design from current directory...")
    # csvs are in current dir
    try:
        design = Design(".", name="ZipCPU", verbose=True)
        print(design.summary())
    except Exception as e:
        print(f"Error loading design: {e}")
        return

    # 2. Querying
    print("\n[2] Querying Interface")
    
    # Check for endpoints (likely registered outputs or flip-flops)
    # We can filter by "Direction=output" and "IsPort=True"
    outputs = design.pins.filter(Direction="output", IsPort=True)
    print(f"  Found {len(outputs)} output ports.")
    
    if not outputs.empty:
        # Show top 5
        print(outputs.head(5)[["Name", "SlackWorst_ns"]])

    # 3. Critical Path
    print("\n[3] Critical Path Analysis")
    paths = design.pins.get_critical_paths(top_k=5)
    print(f"  Found {len(paths)} critical paths:")
    for i, p in enumerate(paths):
        print(f"    {i+1}. Slack: {p['slack']:.4f}ns | Start: {p['startpoint']} -> End: {p['endpoint']}")

    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)

if __name__ == "__main__":
    main()
