// OpenSTA, Static Timing Analyzer
// CSV Writer SWIG Interface
// Provides Tcl bindings for CSV export functions

%module csvWriter

%{
#include "csv/csvWriter.hh"
#include "Sta.hh"

using namespace sta;
%}

%inline %{

void
write_network_nodes_cmd(const char *filename, bool use_internal_nodes)
{
  Sta *sta = Sta::sta();
  writeNetworkNodes(sta, filename, use_internal_nodes);
}

void
write_network_arcs_cmd(const char *filename, bool use_internal_nodes)
{
  Sta *sta = Sta::sta();
  writeNetworkArcs(sta, filename, use_internal_nodes);
}

void
write_pin_properties_cmd(const char *filename, const char *spef_file)
{
  Sta *sta = Sta::sta();
  std::string spef = spef_file ? spef_file : "";
  writePinPropertiesCsv(sta, filename, spef);
}

void
write_cell_properties_cmd(const char *filename)
{
  Sta *sta = Sta::sta();
  writeCellPropertiesCsv(sta, filename);
}

// Benchmark function: extracts only Tcl get_property accessible fields
void
write_instance_properties_benchmark_cmd(const char *filename)
{
  Sta *sta = Sta::sta();
  writeInstancePropertiesBenchmark(sta, filename);
}

%} // inline
