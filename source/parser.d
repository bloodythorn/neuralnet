module parser;

import std.conv;
import std.stdio;
import std.string;

/* IN-Parsing Struct */
struct Digit {
  static const ulong WIDTH = 8;
  static const ulong HEIGHT = 9;
  bool[HEIGHT][WIDTH] m_rep;
  ulong m_value = 0;
  double[] toInput(){
    double[] output;
    foreach(a; m_rep) foreach(b; a)
      if(b) output ~= 0.99;
      else output ~= 0.01;
    return output;
  }
  string toString() {
      string output;
      output ~= to!string(m_value) ~ "\n";
      foreach(a; m_rep)
        output ~= to!string(a) ~ "\n";
      output = output[0..$-1];
      return output;
  }
  bool test() {
    bool output = false;
    foreach(a; m_rep) foreach(b; a) output |= b;
    return output;
  }
}

Digit[] parseInput(string p_file) {
  Digit[] output;

  File file = File(p_file);
  if(!file.isOpen) return output;

  while(!file.eof){
    Digit temp = parseDigit(file);
    if(temp.test)
      output ~= temp;
    else return output;
  }

  return output;
}

Digit parseDigit(File p_file) {

  /* Setup */
  Digit output;
  string line;
  ulong row = 0;
  ulong col= 0;
  /* Iterate through each line */
  while (!p_file.eof) {
    line = strip(p_file.readln);
    if (row >= 0 && row < Digit.HEIGHT) { /* Normal Row */
      if (line.length == Digit.WIDTH) { /* Normal Row Size */
        while (line.length != 0) {
          switch (line[0]) {
            case '0': {
              output.m_rep[col++][row] = false;
              break;
            }
            case '1': {
              output.m_rep[col++][row] = true;
              break;
            }
            default: {
              output = Digit();
              return output;
            }
          }
          line = line[1..$];
        }
        col = 0;
        row++;
      } else {
        output = Digit();
        return output;
      }
    }
    else if (row == Digit.HEIGHT) { /* Read Number at end */
      if (line.length == 1) {
        ulong value = parse!ulong(line);
        output.m_value = value;
        if(line.length > 0) {
          output = Digit();
          return output;
        }
      }
      else {
        output = Digit();
        return output;
      }
      row++;
    }
    else {
      /* This can be only whitespace */
      if (line.length != 0) {
        output = Digit();
        return output;
      }
      else break;
    }
  }
  if (row != 10 || col != 0) {
    output = Digit();
    return output;
  }
  return output;
}

double[] getTarget(ulong p_digit, ulong p_size) {
  double[] output;
  for(uint i = 0; i < p_size;++i)
    if(i == p_digit) output ~= 0.99;
    else output ~= 0.01;
  return output;
}