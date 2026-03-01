/*
module top (in1, in2, clk1, clk2, clk3, out);
  input in1, in2, clk1, clk2, clk3;
  output out;
  wire r1q, r2q, u1z, u2z;

  DFF_X1 r1 (.D(in1), .CK(clk1), .Q(r1q));
  DFF_X1 r2 (.D(in2), .CK(clk2), .Q(r2q));
  BUF_X1 u1 (.A(r2q), .Z(u1z));
  AND2_X1 u2 (.A1(r1q), .A2(u1z), .ZN(u2z));
  DFF_X1 r3 (.D(u2z), .CK(clk3), .Q(out));
endmodule // top
*/



module top (in1, clk1, clk2, sel, op1);
input in1, clk1, clk2, sel;
output op1;
BUF_X1 b1 (.A(in1), .Z(w1));
BUF_X1 b5 (.A(w6), .Z(w7));
block block (.in(w1), .clk(w7), .op(w5));
BUF_X1 b4 (.A(w5), .Z(op1));
MUX2_X1 m1 (.A(clk1), .B(clk2), .S(sel), .Z(w6));
endmodule

module block (in, clk, op);
input in, clk;
output op;

BUF_X1 b2 (.A(in), .Z(w2));
DFF_X1 r1 (.D(w2), .CK(clk), .Q(w3));
BUF_X1 b3 (.A(w3), .Z(w4));
DFF_X1 r2 (.D(w4), .CK(w8), .Q(op));
BUF_X1 b6 (.A(clk), .Z(w8));
endmodule
