#!/usr/bin/env tclsh
#==============================================================================
# BENCHMARK: C++ writeInstancePropertiesBenchmark via Tcl command
#
# This script calls the native C++ extraction function.
# Usage: source this file in OpenSTA after loading your design
#==============================================================================

# Output file
set output_file "instance_properties_cpp.csv"

puts "============================================================"
puts "BENCHMARK: C++ Native Instance Property Extraction"
puts "============================================================"
puts ""

# Start timer
set start_time [clock microseconds]

# Call the C++ function (you need to register this command)
write_instance_properties_benchmark $output_file

# End timer
set end_time [clock microseconds]
set elapsed_ms [expr {($end_time - $start_time) / 1000.0}]
set elapsed_s [expr {$elapsed_ms / 1000.0}]

# Count lines in output (instances)
set fp [open $output_file "r"]
set line_count -1  ;# Subtract header
while {[gets $fp line] >= 0} {
    incr line_count
}
close $fp

puts ""
puts "============================================================"
puts "RESULTS"
puts "============================================================"
puts "  Instances extracted: $line_count"
puts "  Output file: $output_file"
puts "  Elapsed time: [format "%.3f" $elapsed_s] seconds ([format "%.1f" $elapsed_ms] ms)"
puts "  Throughput: [format "%.0f" [expr {$line_count / $elapsed_s}]] instances/second"
puts "============================================================"
