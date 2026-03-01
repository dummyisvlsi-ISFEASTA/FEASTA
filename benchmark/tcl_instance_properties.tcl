#!/usr/bin/env tclsh
#==============================================================================
# BENCHMARK: Tcl get_property vs C++ writeInstancePropertiesBenchmark
#
# This script extracts instance properties using pure Tcl get_property calls
# to compare against the native C++ extraction.
#
# Usage: source this file in OpenSTA after loading your design
#==============================================================================

# Output file
set output_file "instance_properties_tcl.csv"

puts "============================================================"
puts "BENCHMARK: Tcl get_property Instance Property Extraction"
puts "============================================================"
puts ""

# Start timer
set start_time [clock microseconds]

# Open output file
set fp [open $output_file "w"]

# Write CSV header (matching C++ function)
puts $fp "full_name,name,ref_name,liberty_cell,is_buffer,is_inverter,is_macro,is_memory,is_clock_gate,is_hierarchical"

# Get all instances
set instance_count 0
set instances [get_cells -hierarchical *]

foreach inst $instances {
    # Skip if not a leaf cell
    if {[get_property $inst is_hierarchical]} {
        continue
    }
    
    # Extract properties using get_property
    set full_name [get_property $inst full_name]
    set name [get_property $inst name]
    set ref_name [get_property $inst ref_name]
    set liberty_cell [get_property $inst liberty_cell]
    
    # Boolean properties
    set is_buffer [get_property $inst is_buffer]
    set is_inverter [get_property $inst is_inverter]
    set is_macro [get_property $inst is_macro]
    set is_memory [get_property $inst is_memory]
    set is_clock_gate [get_property $inst is_clock_gate]
    set is_hierarchical [get_property $inst is_hierarchical]
    
    # Convert to true/false strings
    set is_buffer_str [expr {$is_buffer ? "true" : "false"}]
    set is_inverter_str [expr {$is_inverter ? "true" : "false"}]
    set is_macro_str [expr {$is_macro ? "true" : "false"}]
    set is_memory_str [expr {$is_memory ? "true" : "false"}]
    set is_clock_gate_str [expr {$is_clock_gate ? "true" : "false"}]
    set is_hierarchical_str [expr {$is_hierarchical ? "true" : "false"}]
    
    # Write CSV row
    puts $fp "$full_name,$name,$ref_name,$liberty_cell,$is_buffer_str,$is_inverter_str,$is_macro_str,$is_memory_str,$is_clock_gate_str,$is_hierarchical_str"
    
    incr instance_count
    
    # Progress indicator
    if {$instance_count % 10000 == 0} {
        puts "  Processed $instance_count instances..."
    }
}

close $fp

# End timer
set end_time [clock microseconds]
set elapsed_ms [expr {($end_time - $start_time) / 1000.0}]
set elapsed_s [expr {$elapsed_ms / 1000.0}]

puts ""
puts "============================================================"
puts "RESULTS"
puts "============================================================"
puts "  Instances extracted: $instance_count"
puts "  Output file: $output_file"
puts "  Elapsed time: [format "%.3f" $elapsed_s] seconds ([format "%.1f" $elapsed_ms] ms)"
puts "  Throughput: [format "%.0f" [expr {$instance_count / $elapsed_s}]] instances/second"
puts "============================================================"
