#!/usr/bin/env tclsh
#==============================================================================
#                    FEASTA BENCHMARK: Tcl vs C++ Speedup
#==============================================================================

#------------------------------------------------------------------------------
#  CONFIGURE YOUR DESIGN HERE (edit these paths)
#------------------------------------------------------------------------------

set LIBERTY_FILE   "examples/sky130hd_tt.lib.gz"      ;# Your .lib or .lib.gz
set VERILOG_FILE   "examples/gcd_sky130hd.v"          ;# Your .v netlist
set TOP_MODULE     "gcd"                              ;# Top module name
set SDC_FILE       "examples/gcd_sky130hd.sdc"        ;# SDC file (or "" to skip)
set SPEF_FILE      ""                                 ;# SPEF file (or "" to skip)

puts "\n╔══════════════════════════════════════════════════════════════╗"
puts "║        FEASTA Benchmark: Tcl vs C++ Native Extraction        ║"
puts "╚══════════════════════════════════════════════════════════════╝\n"

# Load design
puts "Loading design..."
puts "  Liberty: $LIBERTY_FILE"
read_liberty $LIBERTY_FILE

puts "  Verilog: $VERILOG_FILE"
read_verilog $VERILOG_FILE
link_design $TOP_MODULE

if {$SDC_FILE ne ""} {
    puts "  SDC: $SDC_FILE"
    read_sdc $SDC_FILE
}

if {$SPEF_FILE ne ""} {
    puts "  SPEF: $SPEF_FILE"
    read_spef $SPEF_FILE
}

puts "\nDesign loaded successfully.\n"

#------------------------------------------------------------------------------
# BENCHMARK 1: Tcl get_property extraction
#------------------------------------------------------------------------------

puts "┌────────────────────────────────────────────────────────────────┐"
puts "│  Running Tcl get_property extraction...                       │"
puts "└────────────────────────────────────────────────────────────────┘"

set tcl_start [clock microseconds]
set fp [open "instance_properties_tcl.csv" "w"]
puts $fp "full_name,name,ref_name,liberty_cell,is_buffer,is_inverter,is_macro,is_memory,is_clock_gate,is_hierarchical"

set tcl_count 0
foreach inst [get_cells -hierarchical *] {
    if {[get_property $inst is_hierarchical]} { continue }
    
    set full_name [get_property $inst full_name]
    set name [get_property $inst name]
    set ref_name [get_property $inst ref_name]
    set lib_cell [get_property $inst liberty_cell]
    
    if {[get_property $inst is_buffer]} { set is_buf "true" } else { set is_buf "false" }
    if {[get_property $inst is_inverter]} { set is_inv "true" } else { set is_inv "false" }
    if {[get_property $inst is_macro]} { set is_mac "true" } else { set is_mac "false" }
    if {[get_property $inst is_memory]} { set is_mem "true" } else { set is_mem "false" }
    if {[get_property $inst is_clock_gate]} { set is_cg "true" } else { set is_cg "false" }
    
    puts $fp "$full_name,$name,$ref_name,$lib_cell,$is_buf,$is_inv,$is_mac,$is_mem,$is_cg,false"
    
    incr tcl_count
}
close $fp
set tcl_time [expr {([clock microseconds] - $tcl_start) / 1000000.0}]

#------------------------------------------------------------------------------
# BENCHMARK 2: C++ native extraction
#------------------------------------------------------------------------------

puts "┌────────────────────────────────────────────────────────────────┐"
puts "│  Running C++ native extraction...                             │"
puts "└────────────────────────────────────────────────────────────────┘"

set cpp_start [clock microseconds]
sta::write_instance_properties_benchmark_cmd "instance_properties_cpp.csv"
set cpp_time [expr {([clock microseconds] - $cpp_start) / 1000000.0}]

# Count C++ output
set fp [open "instance_properties_cpp.csv" "r"]
set cpp_count -1
while {[gets $fp line] >= 0} { incr cpp_count }
close $fp

#------------------------------------------------------------------------------
# RESULTS
#------------------------------------------------------------------------------

set speedup [expr {$tcl_time / $cpp_time}]

puts ""
puts "╔══════════════════════════════════════════════════════════════╗"
puts "║                         RESULTS                              ║"
puts "╠══════════════════════════════════════════════════════════════╣"
puts [format "║  Design:      %-44s ║" $TOP_MODULE]
puts [format "║  Instances:   %-44d ║" $tcl_count]
puts "╠══════════════════════════════════════════════════════════════╣"
puts [format "║  Tcl Time:    %.3f sec                                       ║" $tcl_time]
puts [format "║  C++ Time:    %.3f sec                                       ║" $cpp_time]
puts "╠══════════════════════════════════════════════════════════════╣"
puts [format "║  🚀 SPEEDUP:  %.1fx faster with C++ extraction               ║" $speedup]
puts "╚══════════════════════════════════════════════════════════════╝"
puts ""

exit
