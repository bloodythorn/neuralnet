module parser;

import std.conv;
import std.stdio;
import std.string;

/* IN-Parsing Struct */
struct Digit {
  static const ulong WIDTH = 8;
  static const ulong HEIGHT = 9;
  bool[WIDTH][HEIGHT] m_rep;
  ulong m_value = 0;
  string toString() {
      string output = to!string(m_value) ~ "|";
      foreach(a; m_rep[0])
        if(a) output ~= "1";
        else output ~= "0";
      output ~= "|\n";
      foreach(a; m_rep[1..$]) {
        output ~= " |";
        foreach(b; a)
          if(b) output ~= "1";
          else output ~= "0";
        output ~= "|\n";
      }
      return output;
  }
  bool test() {
    bool output = false;
    foreach(a; m_rep) foreach(b; a) output |= b;
    return output;
  }
  const double[] getInput() {
    double[] output;
    foreach(a; m_rep) foreach(b; a)
      if(b) output ~= 0.99;
      else output ~= 0.01;
    return output;
  }
  const double[] getTarget(uint p_size) {
    double[] output;
    for(uint i = 0; i < p_size;++i)
      if(i == m_value) output ~= 0.99;
      else output ~= 0.01;
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
  uint row = 0;
  uint col= 0;
  /* Iterate through each line */
  while (!p_file.eof) {
    line = strip(p_file.readln);
    if (row >= 0 && row < Digit.HEIGHT) { /* Normal Row */
      if (line.length == Digit.WIDTH) { /* Normal Row Size */
        while (line.length != 0) {
          switch (line[0]) {
            case '0': {
              output.m_rep[row][col++] = false;
              break;
            }
            case '1': {
              output.m_rep[row][col++] = true;
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
