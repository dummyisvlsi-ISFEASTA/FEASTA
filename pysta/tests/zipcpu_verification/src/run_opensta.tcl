# Example: Run FEASTA extraction on ZipCPU design
# NOTE: Update paths below to point to your local SAED14nm PDK and design files

# Read liberty files (SAED14nm PDK)
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/base/saed14hvt_base_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/cg/saed14hvt_cg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/dlvl/saed14hvt_dlvl_ff0p715v125c_i0p715v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/dlvl/saed14hvt_dlvl_ff0p715v125c_i0p88v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/iso/saed14hvt_iso_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/pg/saed14hvt_pg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/ret/saed14hvt_ret_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_HVT/liberty/nldm/ulvl/saed14hvt_ulvl_ff0p715v125c_i0p715v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/base/saed14lvt_base_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/cg/saed14lvt_cg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/dlvl/saed14lvt_dlvl_ff0p715v125c_i0p715v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/dlvl/saed14lvt_dlvl_ff0p715v125c_i0p88v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/iso/saed14lvt_iso_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/pg/saed14lvt_pg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/ret/saed14lvt_ret_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_LVT/liberty/nldm/ulvl/saed14lvt_ulvl_ff0p715v125c_i0p715v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/base/saed14rvt_base_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/cg/saed14rvt_cg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/dlvl/saed14rvt_dlvl_ff0p715v125c_i0p715v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/dlvl/saed14rvt_dlvl_ff0p715v125c_i0p88v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/iso/saed14rvt_iso_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/pg/saed14rvt_pg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/ret/saed14rvt_ret_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_RVT/liberty/nldm/ulvl/saed14rvt_ulvl_ff0p715v125c_i0p715v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/base/saed14slvt_base_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/cg/saed14slvt_cg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/dlvl/saed14slvt_dlvl_ff0p715v125c_i0p715v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/dlvl/saed14slvt_dlvl_ff0p715v125c_i0p88v.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/iso/saed14slvt_iso_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/pg/saed14slvt_pg_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/ret/saed14slvt_ret_ff0p715v125c.lib"
# read_liberty "<PATH_TO_PDK>/SAED14nm_EDK_STD_SLVT/liberty/nldm/ulvl/saed14slvt_ulvl_ff0p715v125c_i0p715v.lib"

# Read design netlist and parasitics
# read_verilog  <PATH_TO_DESIGN>/netlist_zipcpu.v
# link_design zipcpu
# read_spef <PATH_TO_DESIGN>/test.maxTLU_125.spef
# report_parasitic_annotation

### Timing constraints
# create_clock -name clk -period 0.5 [get_ports i_clk]
# create_clock -name vclk -period 0.5
# set_propagated_clock [get_clocks]
# set_input_delay 0.15 -clock vclk [get_ports -filter "direction == input"]
# set_output_delay 0.15 -clock vclk [get_ports -filter "direction == output"]
# set_clock_uncertainty 0.10 [get_clocks clk] -setup
# set_clock_uncertainty 0.15 [get_clocks clk] -hold
# set_clock_transition 0.05 -rise [get_clocks clk]
# set_clock_transition 0.075 -fall [get_clocks clk]
# set_timing_derate -late 1.1
# set_timing_derate -early 0.9
# set_clock_latency 0.12 -source [get_clocks "clk"]

### FEASTA extraction commands
# dump_network_nodes network_nodes.csv
# dump_network_arcs network_arcs.csv
# dump_cell_properties cell_properties.csv
# dump_pin_properties pin_properties.csv <PATH_TO_DESIGN>/test.maxTLU_125.spef
