# Load design and run benchmark
# This script loads gcd_sky130hd and runs the Tcl vs C++ benchmark

puts "Loading gcd_sky130hd design..."

# Read liberty
read_liberty examples/sky130hd_tt.lib.gz

# Read verilog
read_verilog examples/gcd_sky130hd.v
link_design gcd

# Read constraints
read_sdc examples/gcd_sky130hd.sdc

# Read SPEF for parasitics  
read_spef examples/gcd_sky130hd.spef

puts "Design loaded. Running benchmark..."

# Now run the benchmark
source benchmark/run_benchmark.tcl

exit
